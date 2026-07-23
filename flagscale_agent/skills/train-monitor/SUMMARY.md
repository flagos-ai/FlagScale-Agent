<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# Train-Monitor — Summary

Monitor running FlagScale training jobs: locate logs, check health, detect anomalies, and parse metrics.

**Load when**: monitoring a running training job, diagnosing training anomalies (NaN loss, OOM, hangs), or needing to find/parse training logs.

Key rule: always use `monitor(output_dir=...)` as primary method — it auto-discovers latest logs and scans stderr. Never use raw find commands (they find old logs). Check stderr first, not stdout.
