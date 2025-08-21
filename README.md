# iperf
Parse iperf/iperf3 log files and visualize throughput per node.

Features
- Detects nodes (remote IPs) and ports automatically
- Separates directions (forward vs reverse)
- Aggregates bitrate per (node, direction, port)
- Creates: 
  1) Bar chart of avg Gbit/s per node, split by direction
  2) Heatmap of Gbit/s by node x port (forward and reverse)
  3) CSV export of the parsed table

# Usage
------
python iperf_log_parser_and_visualizer.py /path/to/iperf_log.txt -o outdir
# Multiple files supported
a) python iperf_log_parser_and_visualizer.py iperf_soc505.txt other.log -o plots

Test quickly with the uploaded file path from our chat:
python iperf_log_parser_and_visualizer.py /mnt/data/iperf_soc505.txt -o ./iperf_plots

Requirements: pandas, matplotlib, numpy (pip install pandas matplotlib numpy)

# Notes
-----
- Works with logs similar to iperf3. It looks for the 10s summary lines containing total
  Transfer/Bitrate and prefers the "sender" line when both sender/receiver appear.
- If only one summary appears, it uses that value.
- It tags direction as "reverse" when the block contains a "Reverse mode" marker; otherwise "forward".
