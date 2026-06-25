# FrameGraph

```python
FrameGraph(params)
```

## Description
A wrapper object to construct a frame graph. The frame graph is a directed graph with coordinate frames as vertices and transformations between frames as edges.
New frames and transformations can be added by calling the add_edge() function. Once all frames are specified the frame graph can be used to find a 'path' from 
a source frame to a target frame. Individual transformations will automatically be chained (and inverted if necessary) to produce the transformation from any 
source to any reachable target frame.

## Import

```python
from juniper.robotics import FrameGraph
```
