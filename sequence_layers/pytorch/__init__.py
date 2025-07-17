# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sequence layers in PyTorch."""

# Import all the main components - will be implemented in phases
from sequence_layers.pytorch.types import *

# Phase 2: Simple layers - IMPLEMENTED
from sequence_layers.pytorch.simple import *

# Phase 3: Dense layers - IMPLEMENTED
from sequence_layers.pytorch.dense import *

# Phase 4: Convolutional layers - IMPLEMENTED
from sequence_layers.pytorch.convolution import *

# Phase 5: Recurrent layers - IMPLEMENTED
from sequence_layers.pytorch.recurrent import *

# Phase 6: Attention layers - IMPLEMENTED
from sequence_layers.pytorch.attention import *

# Phase 7: Normalization layers - IMPLEMENTED
from sequence_layers.pytorch.normalization import *

# Phase 8: Pooling layers - IMPLEMENTED
from sequence_layers.pytorch.pooling import *

# Phase 9: DSP layers - IMPLEMENTED
from sequence_layers.pytorch.dsp import *

# Phase 10: Combinators - IMPLEMENTED
from sequence_layers.pytorch.combinators import *

# Phase 11: Specialized layers - IMPLEMENTED
from sequence_layers.pytorch.position import *
from sequence_layers.pytorch.time_varying import *
from sequence_layers.pytorch.conditioning import *

# TODO: Add imports as we implement each module
# from sequence_layers.pytorch.meta import * 