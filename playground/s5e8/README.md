# Bank Term Deposit Subscription Prediction

Start the Prefect server with the following command:

```commandline
$env:PREFECT_API_URL='http://localhost:4200'


docker run -p 4200:4200 -d --rm prefecthq/prefect:3-python3.12 prefect server start --host 0.0.0.0

prefect profile create local
prefect profile use local
prefect config view
```

Run the deployment with the command below:

```commandline
prefect deployment run 'main/bank_subscription_deployment'
```

