for /l %%x in (4, 1, 3000) do (
    echo %%x
    python clustering_kmeans.py %%x
)
