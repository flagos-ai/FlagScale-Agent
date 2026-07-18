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

# Precision Alignment — Summary

Systematically align training precision between reference and target systems using progressive 6-level elimination.

**Load when**: verifying numerical alignment after model porting, comparing loss curves between framework versions, or diagnosing training divergence across hardware.

Three scenarios: Model Migration (native→FlagScale), Internal Iteration (self-regression), Hardware Migration (NVIDIA→new hardware). Six levels: structure → hyperparams → data → init → loss/eval → forward/backward. Each level eliminates one category of variables.
