"""gmapnet — learning on the darts of generalized maps (G-maps).

    gmap.py       triangle mesh -> 2-G-map (darts + involutions), orbits, orientability
    features.py   geometric features carried by orbits (corner angle, dihedral, ...)
    dartconv.py   DartConv: convolution over darts, neighbourhood = involutions
    orbitconv.py  OrbitDartConv: DartConv + orbit pooling (mean, dispersion) + classifier
    data.py       SHREC-16 loading as G-maps, validation split
"""
