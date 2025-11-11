pipeline {
    agent any
    
    environment {
        // Global environment variables
        DOCKER_REGISTRY = 'your-docker-registry'  // e.g., 'docker.io/yourusername'
        DOCKER_CREDENTIALS_ID = 'docker-credentials'
        SONARQUBE_SCANNER_HOME = tool 'SonarQubeScanner'
        SONARQUBE_SERVER = 'SonarQube'  // Name of the SonarQube server configured in Jenkins
        KUBECONFIG = credentials('kubeconfig')  // For Kubernetes deployments
    }
    
    stages {
        // Stage 1: Checkout
        stage('Checkout') {
            steps {
                script {
                    // Define repository URL (replace with your actual GitHub repository URL)
                    def repoUrl = 'https://github.com/yourusername/your-repo.git'
                    
                    // Checkout the repository
                    checkout([
                        $class: 'GitSCM',
                        branches: [[name: '*/main']],  // or '*/master' depending on your default branch
                        extensions: [],
                        userRemoteConfigs: [[
                            credentialsId: 'github-credentials',  // Create this in Jenkins credentials
                            url: repoUrl
                        ]]
                    ])
                    
                    // Get the current Git commit hash for versioning
                    env.GIT_COMMIT_HASH = sh(script: 'git rev-parse --short HEAD', returnStdout: true).trim()
                    env.APP_VERSION = "${env.BUILD_NUMBER}-${env.GIT_COMMIT_HASH}"
                    
                    // Print repository info for debugging
                    sh 'git remote -v'
                    sh 'git branch -v'
                }
            }
        }
        
        // Stage 2: Build
        stage('Build') {
            steps {
                script {
                    // For Java projects
                    if (fileExists('pom.xml')) {
                        sh 'mvn clean package -DskipTests'
                    } 
                    // For Node.js projects
                    else if (fileExists('package.json')) {
                        sh 'npm install'
                        sh 'npm run build'
                    }
                    // For Python projects
                    else if (fileExists('requirements.txt')) {
                        sh 'python -m pip install --upgrade pip'
                        sh 'pip install -r requirements.txt'
                    }
                }
            }
        }
        
        // Stage 3: Unit Tests
        stage('Unit Tests') {
            steps {
                script {
                    // For Java projects
                    if (fileExists('pom.xml')) {
                        sh 'mvn test'
                        junit '**/target/surefire-reports/*.xml'
                    } 
                    // For Node.js projects
                    else if (fileExists('package.json')) {
                        sh 'npm test'
                        junit 'test-results.xml'  // Update based on your test reporter
                    }
                    // For Python projects
                    else if (fileExists('setup.py') || fileExists('pytest.ini')) {
                        sh 'pytest --junitxml=test-results.xml'
                        junit 'test-results.xml'
                    }
                }
            }
            post {
                always {
                    // Archive test results
                    junit allowEmptyResults: true, testResults: '**/test-results.xml'
                }
            }
        }
        
        // Stage 4: SonarQube Analysis
        stage('SonarQube Analysis') {
            steps {
                script {
                    withSonarQubeEnv('SonarQube') {
                        // For Java projects
                        if (fileExists('pom.xml')) {
                            sh 'mvn sonar:sonar -Dsonar.projectVersion=${APP_VERSION}'
                        } 
                        // For other languages, use the sonar-scanner
                        else {
                            sh "${SONARQUBE_SCANNER_HOME}/bin/sonar-scanner " +
                                "-Dsonar.projectKey=${JOB_NAME} " +
                                "-Dsonar.projectName=${JOB_NAME} " +
                                "-Dsonar.projectVersion=${APP_VERSION} " +
                                "-Dsonar.sources=. " +
                                "-Dsonar.sourceEncoding=UTF-8"
                        }
                    }
                }
            }
        }
        
        // Stage 5: Quality Gate
        stage('Quality Gate') {
            steps {
                timeout(time: 5, unit: 'MINUTES') {
                    script {
                        def qg = waitForQualityGate()
                        if (qg.status != 'OK') {
                            error "Pipeline aborted due to quality gate failure: ${qg.status}"
                        }
                    }
                }
            }
        }
        
        // Stage 6: Build Docker Image
        stage('Build Docker Image') {
            steps {
                script {
                    def dockerImage = docker.build("${DOCKER_REGISTRY}/heart-disease-app:${env.APP_VERSION}")
                    docker.withRegistry('', DOCKER_CREDENTIALS_ID) {
                        dockerImage.push()
                    }
                }
            }
        }
        
        // Stage 7: Push to Registry
        stage('Push to Registry') {
            steps {
                script {
                    docker.withRegistry('', DOCKER_CREDENTIALS_ID) {
                        docker.image("${DOCKER_REGISTRY}/heart-disease-app:${env.APP_VERSION}").push()
                    }
                }
            }
        }
        
        // Stage 8: Deploy to Dev
        stage('Deploy to Dev') {
            steps {
                script {
                    // Example using kubectl for Kubernetes deployment
                    sh """
                        kubectl config use-context dev-cluster
                        kubectl set image deployment/heart-disease-app heart-disease-app=${DOCKER_REGISTRY}/heart-disease-app:${env.APP_VERSION} --record
                    """
                }
            }
        }
        
        // Stage 9: Integration Tests
        stage('Integration Tests') {
            steps {
                script {
                    // Example: Run integration tests against the dev environment
                    if (fileExists('integration-tests')) {
                        dir('integration-tests') {
                            sh 'npm install'
                            sh 'npm run test:integration'
                        }
                    } else {
                        echo 'No integration tests found. Skipping...'
                    }
                }
            }
        }
        
        // Stage 10: Performance Tests
        stage('Performance Tests') {
            steps {
                script {
                    // Example using JMeter for performance testing
                    if (fileExists('performance-tests')) {
                        dir('performance-tests') {
                            sh 'jmeter -n -t load_test.jmx -l test_results.jtl'
                            // Archive performance test results
                            perfReport '**/*.jtl'
                        }
                    } else {
                        echo 'No performance tests found. Skipping...'
                    }
                }
            }
        }
        
        // Stage 11: Deploy to Staging
        stage('Deploy to Staging') {
            when {
                // Only deploy to staging from the main branch or release branches
                anyOf {
                    branch 'main'
                    branch 'release/*'
                }
            }
            steps {
                script {
                    sh """
                        kubectl config use-context staging-cluster
                        kubectl set image deployment/heart-disease-app heart-disease-app=${DOCKER_REGISTRY}/heart-disease-app:${env.APP_VERSION} --record
                    """
                }
            }
        }
        
        // Stage 12: Selenium Tests
        stage('Selenium Tests') {
            when {
                // Only run Selenium tests in the staging environment
                anyOf {
                    branch 'main'
                    branch 'release/*'
                }
            }
            steps {
                script {
                    // Example: Run Selenium tests against the staging environment
                    if (fileExists('e2e-tests')) {
                        dir('e2e-tests') {
                            sh 'npm install'
                            sh 'npm run test:e2e'
                        }
                    } else {
                        echo 'No Selenium tests found. Skipping...'
                    }
                }
            }
        }
        
        // Stage 13: Deploy to Production
        stage('Deploy to Production') {
            when {
                // Only deploy to production from the main branch with approval
                branch 'main'
            }
            steps {
                script {
                    // Manual approval step for production deployment
                    timeout(time: 1, unit: 'HOURS') {
                        input message: 'Deploy to production?', ok: 'Deploy'
                    }
                    
                    // Deploy to production
                    sh """
                        kubectl config use-context production-cluster
                        kubectl set image deployment/heart-disease-app heart-disease-app=${DOCKER_REGISTRY}/heart-disease-app:${env.APP_VERSION} --record
                    """
                    
                    // Send deployment notification
                    slackSend (color: 'good', message: "Successfully deployed ${JOB_NAME} - ${env.APP_VERSION} to production!")
                }
            }
        }
    }
    
    post {
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            // Send failure notification
            slackSend (color: 'danger', message: "${JOB_NAME} - Build #${BUILD_NUMBER} failed. See ${BUILD_URL}")
        }
        cleanup {
            // Clean up workspace after build
            cleanWs()
        }
    }
}
