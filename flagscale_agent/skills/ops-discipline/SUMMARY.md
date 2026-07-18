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

# Operational Discipline — Summary

General operational rules for FlagScale infrastructure work: reading strategy, shell safety, environment awareness, and root cause diagnosis.

**Load when**: starting infrastructure work on a new server, debugging shell/environment issues, or needing structured diagnosis methodology.

Key rules: read complete files before implementing, never run same command twice, use conda run (not activate), check stderr first for errors, verify everything after install. For training-specific operations, use train-run skill instead.
