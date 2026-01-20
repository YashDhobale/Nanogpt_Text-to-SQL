pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                git 'https://github.com/YashDhobale/Nanogpt_Text-to-SQL.git'
            }
        }

        stage('Setup Python') {
            steps {
                sh '''
                python3 -m venv venv
                . venv/bin/activate
                pip install -r requirements.txt
                '''
            }
        }

        stage('Backend CI Check') {
            steps {
                sh '''
                . venv/bin/activate
                python - <<EOF
                import sys
                sys.path.append("src")
                from backend_updated import gen_sql
                sql = gen_sql("show users")
                assert "select" in sql.lower()
                print("Backend CI passed")
                EOF
                '''
            }
        }

        stage('Deploy (Optional)') {
            when {
                branch 'br1'
            }
            steps {
                sshagent(credentials: ['ec2-ssh-key']) {
                    sh '''
                    ssh -o StrictHostKeyChecking=no ubuntu@13.62.64.74 << 'EOF'

                        echo "Stopping existing nanogpt container if running..."
                        docker stop nanogpt || true

                        echo "Starting new nanogpt container..."
                        docker run -d \
                            --name nanogpt-new \
                            --restart unless-stopped \
                            -p 8000:8000 \
                            445876755000.dkr.ecr.eu-north-1.amazonaws.com/cot-hallucinator:latest

                        echo "Deployment completed successfully"
                    EOF
                    '''
                }
            }
    }

    post {
        success {
            echo "Pipeline completed successfully"
        }
        failure {
            echo "Pipeline failed"
        }
    }
}
}