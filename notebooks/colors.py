from matplotlib.colors import LinearSegmentedColormap

DISCRETE_COLORS = [
    "#6ad2e2",
    "#928bde",
    "#ee6541",
    "#e18600",
    "#076d7e",
    "#08312A",
    "#00E47C",
    "#E5E3DE",
    "#ffd03d",
]

DISCRETE_COLORS_BRIGHT = [
    "#00D0FF",
    "#6048FF",
    "#FF3C00",
    "#FF7F00",
    "#00A651",
    "#FF2DA6",
]

DISCRETE_COLORS_ORIG = [
    "#ee6541",
    "#928bde",
    "#00D0FF",
    "#ffd03d",
    "#00A651",
    "#076d7e",
    "#08312A",
]

bi_discrete_cmap = LinearSegmentedColormap.from_list(
    "BI_Discrete",
    DISCRETE_COLORS,
    N=9,
)

bi_continuous_cmap = LinearSegmentedColormap.from_list(
    "BI_Continuous",
    ["#E5E3DE", "#6ad2e2", "#076d7e", "#08312A"],
)


bi_diverging_cmap = LinearSegmentedColormap.from_list(
    "BI_Continuous_Bright",
    [
        "#6048FF",
        "#00D0FF",
        "#FF3C00",
    ],
    N=30,
)


bi_diverging_cmap_red_green = LinearSegmentedColormap.from_list(
    "bi_diverging_cmap_red_green",
    [
        (0.0, "#00A651"),
        (0.5, "#E5E3DE"),
        (1.0, "#FF3C00"),
    ],
)


bi_diverging_onesided = LinearSegmentedColormap.from_list(
    "bi_diverging_onesided",
    [
        (0, "#E5E3DE"),
        (1.0, "#FF3C00"),
    ],
)

cmap = LinearSegmentedColormap.from_list(
    "bi_diverging_cmap_red_green",
    [
        (0.0, "#00A651"),
        (0.2, "#8CD1A4"),
        (0.8, "#E5E3DE"),
        (1.0, "#E05A47"),
    ],
)
