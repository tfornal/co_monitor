import json

diagnostic_coordinates = {
    "detector": {
        "element": {
            "B": {
                "orientation vector": [1.924, 0.199, 0.085],
                "vertex": {
                    "A": [-6259.727, 6802.972, -589.214],
                    "B": [-6257.067, 6776.564, -587.463],
                    "C": [-6257.406, 6776.972, -580.784],
                    "D": [-6260.065, 6803.381, -582.53],
                },
            },
            "C": {
                "orientation vector": [0.34, 1.90, 0.098],
                "vertex": {
                    "A": [-5438.446, 7589.626, 2.706],
                    "B": [-5412.307, 7584.961, 1.106],
                    "C": [-5412.643, 7585.367, -5.573],
                    "D": [-5438.782, 7590.032, -3.98],
                },
            },
            "N": {
                "orientation vector": [0.594, 1.842, -0.083],
                "vertex": {
                    "A": [-5408.529, 7278.827, -602.019],
                    "B": [-5433.813, 7286.898, -603.795],
                    "C": [-5434.151, 7287.307, -597.116],
                    "D": [-5408.868, 7279.236, -595.3],
                },
            },
            "O": {
                "orientation vector": [1.3478, 1.39, 0.017],
                "vertex": {
                    "A": [-5901.357, 6489.582, -67.661],
                    "B": [-5920.415, 6508.022, -65.581],
                    "C": [-5920.752, 6508.427, -72.261],
                    "D": [-5901.693, 6489.988, -74],
                },
            },
        }
    },
    "chamber orientation vector": {
        "upper": [-58.553, 70.592, 7.236],
        "bottom": [-7.637, 9.207, -0.95],
    },
    "collimator": {
        "element": {
            "B": {
                "vector front-back": [-50.837, 61.29, -6.328],
                "top closing side": {
                    "vector_top_bottom": [0.05, -0.061, -0.996],
                    "vertex": {
                        "A1": [-5495.041, 6707.251, -530.817],
                        "B1": [-5494.991, 6707.190, -531.813],
                        "A2": [-5418.072, 6771.093, -530.817],
                        "B2": [-5418.022, 6771.032, -531.813],
                    },
                },
                "bottom closing side": {
                    "vector_top_bottom": [-0.05, 0.061, 0.996],
                    "vertex": {
                        "A1": [-5417.111, 6769.934, -549.757],
                        "B1": [-5417.162, 6769.995, -548.760],
                        "A2": [-5494.080, 6706.093, -549.757],
                        "B2": [-5494.131, 6706.154, -548.760],
                    },
                },
            },
            "C": {
                "vector front-back": [-50.839, 61.293, 6.282],
                "top closing side": {
                    "vector_top_bottom": [0.05, -0.06, 0.996],
                    "vertex": {
                        "A1": [-5416.597, 6769.376, -59.739],
                        "B1": [-5416.547, 6769.316, -58.743],
                        "A2": [-5493.566, 6705.535, -59.739],
                        "B2": [-5493.516, 6705.474, -58.743],
                    },
                },
                "bottom closing side": {
                    "vector_top_bottom": [-0.05, 0.061, -0.997],
                    "vertex": {
                        "A1": [-5492.612, 6704.384, -40.798],
                        "B1": [-5492.662, 6704.445, -41.795],
                        "A2": [-5415.643, 6768.226, -40.798],
                        "B2": [-5415.693, 6768.587, -41.795],
                    },
                },
            },
            "N": {
                "vector front-back": [-50.836, 61.29, -6.328],
                "top closing side": {
                    "vector_top_bottom": [0.05, -0.061, -0.997],
                    "vertex": {
                        "A1": [-5493.574, 6705.483, -559.725],
                        "B1": [-5493.524, 6705.422, -560.722],
                        "A2": [-5416.606, 6769.325, -559.725],
                        "B2": [-5416.555, 6769.264, -560.722],
                    },
                },
                "bottom closing side": {
                    "vector_top_bottom": [-0.05, 0.061, 0.997],
                    "vertex": {
                        "A1": [-5415.645, 6768.166, -578.666],
                        "B1": [-5415.695, 6768.227, -577.669],
                        "A2": [-5492.613, 6704.324, -578.666],
                        "B2": [-5492.664, 6704.385, -577.669],
                    },
                },
            },
            "O": {
                "vector front-back": [-50.839, 61.292, 6.283],
                "top closing side": {
                    "vector_top_bottom": [0.05, -0.061, 0.997],
                    "vertex": {
                        "A1": [-5418.053, 6771.132, -88.65],
                        "B1": [-5418.003, 6771.071, -87.653],
                        "A2": [-5495.022, 6707.290, -88.65],
                        "B2": [-5494.972, 6707.230, -87.653],
                    },
                },
                "bottom closing side": {
                    "vector_top_bottom": [-0.05, 0.06, -0.996],
                    "vertex": {
                        "A1": [-5494.068, 6706.140, -69.709],
                        "B1": [-5494.118, 6706.200, -70.705],
                        "A2": [-5417.099, 6769.982, -69.709],
                        "B2": [-5417.150, 6770.042, -70.705],
                    },
                },
            },
        }
    },
    "dispersive element": {
        "element": {
            "B": {
                "AOI": 29.07,
                "max reflectivity": 25.5,
                "crystal central point": [-5558.849, 6862.497, -553.08],
                "radius central point": [-6097.687, 5549.987, -500.133],
                "vertex": {
                    "A": [-5596.52, 6877.643, -545.887],
                    "B": [-5522.616, 6847.528, -540.295],
                    "C": [-5521.605, 6846.308, -560.232],
                    "D": [-5595.509, 6876.423, -565.824],
                },
            },
            "C": {
                "AOI": 24.94,
                "max reflectivity": 25.7,
                "crystal central point": [-5557.455, 6860.881, -38.06],
                "radius central point": [-3888.684, 7301.849, -95.335],
                "vertex": {
                    "A": [-5567.611, 6900.203, -45.192],
                    "B": [-5547.408, 6823.007, -50.90],
                    "C": [-5546.404, 6821.796, -30.959],
                    "D": [-5566.607, 6898.993, -25.254]
                    
                },
            },
            "N": {
                "AOI": 29.71,
                "max reflectivity": 6.3,
                "crystal central point": [-5557.392, 6860.741, -581.79],
                "radius central point": [-4695.14, 7034.455, -548.667],
                "vertex": {
                    "A": [-5563.773, 6899.448, -594.513],
                    "B": [-5548.218, 6821.173, -588.936],
                    "C": [-5549.23, 6822.393, -568.999],
                    "D": [-5564.785, 6900.667, -574.57]
                    
                },
            },
            "O": {
                "AOI": 46.86,
                "max reflectivity": 5,
                "crystal central point": [-5558.888, 6862.606, -66.472],
                "radius central point": [-5588.096, 6184.396, -106.185],
                "vertex": {
                    "A": [-5598.349, 6862.422, -54.464],
                    "B": [-5518.524, 6859.231, -58.679],
                    "C": [-5519.528, 6860.442, -78.617],
                    "D": [-5599.353, 6863.353, -74.4],
                },
            },
        },
    },
    "ECRH shield": {
        "upper chamber": {
            "1st shield": {
                "radius": 67.5,
                "central point": [-5366.156, 6630.245, -79.757],
                "orientation vector": [-1.909, 2.301, 0.236],
                "edge 1": [-5395.061, 6612.241, -138.019],
                "edge 2": [-5337.251, 6648.249, -21.495],
            },
            "2nd shield": {
                "radius": 67.5,
                "central point": [-5223.943, 6458.79, -97.331],
                "orientation vector": [-1.273, 1.535, 0.157],
                "edge 1": [-5220.554, 6454.704, -30.04],
                "edge 2": [-5227.332, 6462.876, -164.622],
            },
        },
        "bottom chamber": {
            "1st shield": {
                "radius": 67.5,
                "central point": [-5201.634, 6431.894, -525.812],
                "orientation vector": [-1.909, 2.302, -0.238],
                "edge 1": [-5198.605, 6427.459, -593.098],
                "edge 2": [-5204.663, 6436.329, -458.526],
            },
            "2nd shield": {
                "radius": 67.5,
                "central point": [-5324.748, 6580.323, -541.138],
                "orientation vector": [-1.273, 1.534, -0.159],
                "edge 1": [-5321.341, 6576.216, -608.282],
                "edge 2": [-5328.155, 6584.43, -473.994],
            },
        },
    },
    "port": {
        "p1": [-4072.79, 4787.42, -219.45],
        "p2": [-3764.65, 4982.96, -219.45],
        "p3": [-4006.47, 4707.47, -518.92],
        "p4": [-3681.3, 4882.47, -518.92],
        "p5": [-3949.091, 4938.0, -38.43],
        "p6": [-3783.0, 4738.0, -699.0],
        "p7": [-4055.0, 4839.0, -99.0],
        "p8": [-4016.0, 4881.0, -55.0],
        "p9": [-3844.0, 4988.0, -86.0],
        "p10": [-3811.0, 4995.0, -123.0],
        "p11": [-3975.0, 4691.0, -588.0],
        "p12": [-3882.0, 4697.0, -676.0],
        "p13": [-3697.0, 4802.0, -658.0],
        "p14": [-3674.5, 4842.0, -603.0],
        "coplanar vertex": [-3693.98024454, 4902.45205254, -527.06278198],
        "orientation vector": [29.048, -35.021, 0.0],
    },
    "plasma": {
        "central point": [-3430.25, 4305.75, -320.75],
    },
}


with open("coordinates.json", "w") as outfile:
    json.dump(diagnostic_coordinates, outfile)
