# Topo-Detect — Summary

Detect hardware topology: GPUs, CPU, memory, RDMA/NICs, storage, and inter-GPU connectivity.

**Load when**: setting up a new server, planning parallelism strategy, or diagnosing multi-node communication issues.

Generates topology report with GPU count/type, NVLink/PCIe topology, RDMA capabilities, and storage layout. Output feeds into parallel-strategy skill for optimal TP/PP/DP selection.
