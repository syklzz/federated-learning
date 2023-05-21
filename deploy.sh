#!/bin/bash
source .env

# Create repositories
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID".dkr.ecr.us-east-1.amazonaws.com

docker tag "$AGGREGATOR_IMAGE" "$AWS_ACCOUNT_ID".dkr.ecr.us-east-1.amazonaws.com/"$AGGREGATOR"
aws ecr create-repository --repository-name "$AGGREGATOR"
docker push "$AWS_ACCOUNT_ID".dkr.ecr.us-east-1.amazonaws.com/"$AGGREGATOR"

docker tag "$COLL1_IMAGE"  "$AWS_ACCOUNT_ID".dkr.ecr.us-east-1.amazonaws.com/"$COLL1"
aws ecr create-repository --repository-name "$COLL1"
docker push "$AWS_ACCOUNT_ID".dkr.ecr.us-east-1.amazonaws.com/"$COLL1"

docker tag "$COLL2_IMAGE" "$AWS_ACCOUNT_ID".dkr.ecr.us-east-1.amazonaws.com/"$COLL2"
aws ecr create-repository --repository-name "$COLL2"
docker push "$AWS_ACCOUNT_ID".dkr.ecr.us-east-1.amazonaws.com/"$COLL2"

# Configure ecs-cli
ecs-cli configure profile \
  --profile-name federated-learning \
  --access-key "$AWS_ACCESS_KEY_ID" \
  --secret-key "$AWS_SECRET_ACCESS_KEY" \
  --session-token "$AWS_SESSION_TOKEN"

ecs-cli configure \
  --cluster federated-learning-cluster \
  --default-launch-type EC2 \
  --region us-east-1 \
  --config-name federated-learning

# Create an ecs-cluster with one ec2-instance
ecs-cli up \
  --keypair vockey \
  --size 1 \
  --instance-type t3.medium \
  --tags project=federated-learning \
  --cluster-config federated-learning \
  --ecs-profile federated-learning \
  --instance-role LabRole \
  --force

### Deploy
ecs-cli compose \
  --project-name federated-learning \
  --file docker-compose.yml \
  --debug service up  \
  --deployment-max-percent 100 \
  --deployment-min-healthy-percent 0 \
  --region us-east-1 \
  --ecs-profile federated-learning \
  --cluster-config federated-learning

# Open port 22 to connect to the ec2 instance
myip="$(dig +short myip.opendns.com @resolver1.opendns.com)"
sg=$(aws ec2 describe-security-groups --filters Name=tag:project,Values=federated-learning | jq '.SecurityGroups[].GroupId')
sg=$(echo $sg | tr -d '"')
aws ec2 authorize-security-group-ingress \
  --group-id $sg \
  --protocol tcp \
  --port 22 \
  --cidr "$myip/32" | jq '.'


### Connect to ec2 instance
# chmod 400 ~/.aws/labsuser.pem
# ssh -i ~/.aws/labsuser.pem ec2-user@54.160.154.81

