# Infer-Env-Setup — Summary

Set up inference environment for vllm-plugin-FL on hardware backends (MetaX, Ascend, Moore Threads, etc.).

**Load when**: setting up a new inference environment, creating Docker containers for plugin testing, installing vLLM/plugin/FlagGems on a new machine, or reconnecting to an existing environment.

**Full cycle**: SSH connect → environment probe → container creation → vLLM CPU-only install → plugin clone + editable install → FlagGems install → verify imports.

**Key principles**:
- All work inside Docker containers (never install on host)
- Fresh workspace isolation per adaptation task
- Local edit → sync → remote test development model
- Pin vLLM version from plugin's pyproject.toml
- Check device occupancy before workloads
- Use --network host for containers

**Hardware backends** (Container Setup sections): MetaX C550 (complete), Ascend 910B (framework), Moore Threads S4000 (framework). New backends use the "Adding a New Backend" template.

**Constraints**: 10 hard constraints covering SSH, Docker (image match, device mount, workspace volume, network host), installation (container-only, CPU-only vLLM, pinned versions, fresh workspace), and device occupancy. 1 soft constraint for tmux usage.
