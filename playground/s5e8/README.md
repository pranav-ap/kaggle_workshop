# Bank Term Deposit Subscription Prediction

Start the Prefect server with the following command:

```commandline
docker run -p 4200:4200 -d --rm prefecthq/prefect:3-python3.12 prefect server start --host 0.0.0.0
```

Run the deployment with the command below:

```commandline
prefect deployment run 'main/bank_subscription_deployment'
```

