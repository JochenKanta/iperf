#!/usr/bin/env python3
"""
Parse iperf/iperf3 log files and visualize throughput per node.

Features
- Detects nodes (remote IPs) and ports automatically
- Separates directions (forward vs reverse)
- Aggregates bitrate per (node, direction, port)
- Creates: 
  1) Bar chart of avg Gbit/s per node, split by direction
  2) Heatmap of Gbit/s by node x port (forward and reverse)
  3) CSV export of the parsed table

Usage
------
python iperf_log_parser_and_visualizer.py /path/to/iperf_log.txt -o outdir
# Multiple files supported
a) python iperf_log_parser_and_visualizer.py iperf_soc505.txt other.log -o plots

Test quickly with the uploaded file path from our chat:
python iperf_log_parser_and_visualizer.py /mnt/data/iperf_soc505.txt -o ./iperf_plots

Requirements: pandas, matplotlib, numpy (pip install pandas matplotlib numpy)

Notes
-----
- Works with logs similar to iperf3. It looks for the 10s summary lines containing total
  Transfer/Bitrate and prefers the "sender" line when both sender/receiver appear.
- If only one summary appears, it uses that value.
- It tags direction as "reverse" when the block contains a "Reverse mode" marker; otherwise "forward".
"""
from __future__ import annotations
import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Regex patterns
# -------------------------
RE_START_BLOCK = re.compile(r"^Connecting to host\s+(?P<host>\S+),\s+port\s+(?P<port>\d+)")
RE_REMOTE_SENDING = re.compile(r"^Reverse mode,\s+remote host\s+(?P<host>\S+)\s+is sending")
RE_LOCAL_CONNECTED = re.compile(r"connected to\s+(?P<host>\S+)\s+port\s+(?P<port>\d+)")
RE_SUMMARY = re.compile(
    r"^\[\s*\d+\]\s+0\.00-\s*\d+\.\d+\s+sec\s+\s*(?P<gbytes>[\d\.]+)\s+GBytes\s+(?P<gbps>[\d\.]+)\s+Gbits/sec(?:\s+\d+)?\s*(?:sender|receiver)?\s*$",
    re.IGNORECASE,
)
RE_SUMMARY_SENDER = re.compile(
    r"^\[\s*\d+\]\s+0\.00-\s*\d+\.\d+\s+sec\s+(?P<gbytes>[\d\.]+)\s+GBytes\s+(?P<gbps>[\d\.]+)\s+Gbits/sec\s+\d*\s*sender\s*$",
    re.IGNORECASE,
)
RE_SUMMARY_RECEIVER = re.compile(
    r"^\[\s*\d+\]\s+0\.00-\s*\d+\.\d+\s+sec\s+(?P<gbytes>[\d\.]+)\s+GBytes\s+(?P<gbps>[\d\.]+)\s+Gbits/sec\s*receiver\s*$",
    re.IGNORECASE,
)
RE_NODE_ANNOUNCE = re.compile(r"Starting\s+multi-port\s+run\s+\d+\s+for\s+node\s+(?P<node>\S+)")
RE_SENDING_TO = re.compile(r"SENDING TO:\s+(?P<node>\S+)")
RE_RECEIVING_FROM = re.compile(r"RECEIVING FROM:\s+(?P<node>\S+)\s+Reverse Mode", re.IGNORECASE)

@dataclass
class Record:
    node: str
    port: int
    direction: str  # 'forward' | 'reverse'
    gbytes: float
    gbps: float
    src_hint: str  # a short context string


def parse_iperf_logs(paths: Iterable[Path]) -> List[Record]:
    records: List[Record] = []
    for path in paths:
        current_node: Optional[str] = None
        reverse_mode: bool = False
        current_port: Optional[int] = None
        block_lines: List[str] = []

        def flush_block():
            nonlocal records, current_node, reverse_mode, current_port, block_lines
            if current_port is None:
                return
            gbytes = None
            gbps = None
            sender_found = None
            receiver_found = None
            generic_found = None
            for line in block_lines:
                m = RE_SUMMARY_SENDER.match(line)
                if m:
                    sender_found = m
                m2 = RE_SUMMARY_RECEIVER.match(line)
                if m2 and receiver_found is None:
                    receiver_found = m2
                m3 = RE_SUMMARY.match(line)
                if m3 and generic_found is None:
                    generic_found = m3
            use = sender_found or generic_found or receiver_found
            if use:
                gbytes = float(use.group('gbytes'))
                gbps = float(use.group('gbps'))
                records.append(
                    Record(
                        node=current_node or "unknown",
                        port=int(current_port),
                        direction="reverse" if reverse_mode else "forward",
                        gbytes=gbytes,
                        gbps=gbps,
                        src_hint=f"{path.name}:port{current_port}:{'rev' if reverse_mode else 'fwd'}",
                    )
                )
            current_port = None
            block_lines = []

        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.rstrip("\n")
                m_node = RE_NODE_ANNOUNCE.search(line) or RE_SENDING_TO.search(line) or RE_RECEIVING_FROM.search(line)
                if m_node:
                    current_node = m_node.group("node")

                if RE_REMOTE_SENDING.search(line):
                    reverse_mode = True
                    mhost = RE_REMOTE_SENDING.search(line)
                    if mhost:
                        current_node = mhost.group("host")

                m_start = RE_START_BLOCK.match(line)
                if m_start:
                    flush_block()
                    current_port = int(m_start.group("port"))
                    current_node = m_start.group("host") or current_node or "unknown"
                    block_lines = []
                    continue

                m_local_conn = RE_LOCAL_CONNECTED.search(line)
                if m_local_conn and current_port is None:
                    current_port = int(m_local_conn.group("port"))
                    current_node = m_local_conn.group("host") or current_node or "unknown"
                    block_lines = []
                    continue

                if current_port is not None:
                    block_lines.append(line)

                if line.strip().startswith("iperf Done."):
                    flush_block()
                    reverse_mode = False

        flush_block()

    return records


def to_dataframe(records: List[Record]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["node", "port", "direction", "gbytes", "gbps", "src_hint"])
    df = pd.DataFrame([r.__dict__ for r in records])
    df["node"] = df["node"].str.strip()
    return df


def plot_bar_per_node(df: pd.DataFrame, outdir: Path) -> Path:
    g = df.groupby(["node", "direction"])['gbps'].mean().reset_index()
    order = g.groupby('node')['gbps'].mean().sort_values(ascending=False).index
    nodes = list(order)
    directions = sorted(g['direction'].unique())
    x = np.arange(len(nodes))
    width = 0.35 if len(directions) == 2 else 0.6

    fig, ax = plt.subplots(figsize=(max(8, len(nodes)*1.2), 5))
    for i, d in enumerate(directions):
        vals = g[g['direction']==d].set_index('node').reindex(nodes)['gbps'].values
        vals = np.nan_to_num(vals)
        ax.bar(x + (i - (len(directions)-1)/2)*width, vals, width, label=d)

    ax.set_xticks(x)
    ax.set_xticklabels(nodes, rotation=30, ha='right')
    ax.set_ylabel('Durchsatz (Gbit/s)')
    ax.set_title('Ø Durchsatz pro Knoten (nach Richtung)')
    ax.legend(title='Richtung')
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    outpath = outdir / 'avg_throughput_per_node.png'
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


def plot_heatmap(df: pd.DataFrame, outdir: Path, direction: str) -> Optional[Path]:
    if direction not in df['direction'].unique():
        return None
    piv = df[df['direction']==direction].pivot_table(index='node', columns='port', values='gbps', aggfunc='mean')
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(max(8, piv.shape[1]*0.6), max(4, piv.shape[0]*0.4)))
    im = ax.imshow(piv.values, aspect='auto')
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns.astype(str), rotation=0)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    ax.set_title(f'Durchsatz-Heatmap (Richtung: {direction}) [Gbit/s]')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Gbit/s')
    ax.set_xlabel('Port')
    ax.set_ylabel('Knoten')
    fig.tight_layout()
    outpath = outdir / f'heatmap_{direction}.png'
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


def main():
    p = argparse.ArgumentParser(description="Parse iperf logs and plot throughput per node")
    p.add_argument('logs', nargs='*', type=Path, help='iperf/iperf3 log file(s)')
    p.add_argument('-o', '--outdir', type=Path, default=Path('./iperf_plots'), help='Output directory')
    p.add_argument('--no-heatmap', action='store_true', help='Skip heatmap generation')
    p.add_argument('--csv', action='store_true', help='Also export parsed table to CSV')
    args = p.parse_args()

    default_log = Path('/mnt/data/iperf_soc505.txt')
    if not args.logs:
        if default_log.exists():
            args.logs = [default_log]
            print(f"Keine Logs übergeben. Nutze Standard: {default_log}")
        else:
            print("Fehler: Keine Logdatei übergeben. Beispiel:\n  python iperf_log_parser_and_visualizer.py /pfad/zu/iperf.log -o ./iperf_plots --csv")
            return

    args.outdir.mkdir(parents=True, exist_ok=True)

    records = parse_iperf_logs(args.logs)
    if not records:
        print("Keine passenden Daten gefunden. Prüfe das Logformat.")
        return

    df = to_dataframe(records)

    if args.csv:
        csv_path = args.outdir / 'iperf_parsed.csv'
        df.to_csv(csv_path, index=False)
        print(f"CSV geschrieben: {csv_path}")

    bar_path = plot_bar_per_node(df, args.outdir)
    print(f"Diagramm geschrieben: {bar_path}")

    if not args.no_heatmap:
        for direction in sorted(df['direction'].unique()):
            hp = plot_heatmap(df, args.outdir, direction)
            if hp:
                print(f"Heatmap geschrieben: {hp}")

    g = df.groupby(['node','direction'])['gbps'].agg(['count','mean','median','min','max']).round(2)
    print("\nZusammenfassung (Gbit/s):")
    print(g)

if __name__ == '__main__':
    main()
