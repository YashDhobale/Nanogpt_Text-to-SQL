pipeline {
    agent any

    environment {
        EC2_USER = "ubuntu"
        EC2_HOST = "13.62.64.74"
        APP_DIR  = "/home/ubuntu/nanogpt-backend"
        IMAGE_NAME = "nanogptnew"
    }

    stages {

        stage('Deploy') {
            when {
                expression {
                    return env.GIT_BRANCH == 'origin/br1'
                }   
            }
            steps {
                sshagent(credentials: ['ec2-ssh-key']) {
                    sh '''
                    ssh -o StrictHostKeyChecking=no ${EC2_USER}@${EC2_HOST} << EOF
                      
                      echo "Going to app directory"
                      cd ${APP_DIR}

                      echo "Pulling latest code"
                      git pull origin br1

                      echo "Building new Docker image"
                      docker build -t ${IMAGE_NAME}:latest .

                      echo "Stopping existing nanogpt container (if running)"
                      docker stop nanogpt || true

                      echo "Starting new nanogpt container"
                      docker run -d \
                        --name nanogptnewcont \
                        --restart unless-stopped \
                        -p 8000:8000 \
                        ${IMAGE_NAME}:latest

                      echo "Deployment from br1 completed successfully"
                    EOF
                    '''
                }
            }
        }
    }
}
