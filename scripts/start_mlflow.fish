 mlflow server \
    --backend-store-uri sqlite:////home/richw/.code/latent-space/mlflow.db \
    --default-artifact-root /home/richw/.code/latent-space/artifacts \
    --host 0.0.0.0 \
    --port 5050 \
    --allowed-hosts "100.121.43.41,100.121.43.41:5050,localhost,localhost:5050" \
    --cors-allowed-origins "http://100.121.43.41:5050,http://localhost:5050"
