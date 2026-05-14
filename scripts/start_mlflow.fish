mlflow server \
    --backend-store-uri sqlite:////home/richw/.code/latent-space/mlflow.db \
    --default-artifact-root mlflow-artifacts:/ \
    --artifacts-destination /home/richw/.code/latent-space/artifacts \
    --host 0.0.0.0 \
    --port 5050 \
    --serve-artifacts \
    --allowed-hosts "172.20.199.236,172.20.199.236:5050,100.100.16.25,100.100.16.25:5050,localhost,localhost:5050" \
    --cors-allowed-origins "http://100.100.16.25:5050,http://localhost:5050,http://172.20.199.236:5050"
