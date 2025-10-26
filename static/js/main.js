/**
 * HeartGuard AI - Main JavaScript File
 * Interactive features and functionality
 */

// Global variables
let predictionForm = null;
let analysisData = null;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

/**
 * Initialize the application
 */
function initializeApp() {
    // Initialize form handling
    initializeForms();
    
    // Initialize animations
    initializeAnimations();
    
    // Initialize tooltips
    initializeTooltips();
    
    // Initialize charts if on analysis page
    if (document.getElementById('featureImportanceChart')) {
        initializeAnalysis();
    }
    
    // Initialize health tips if on about page
    if (document.getElementById('healthTips')) {
        loadHealthTips();
    }
}

/**
 * Initialize form handling
 */
function initializeForms() {
    predictionForm = document.getElementById('predictionForm');
    
    if (predictionForm) {
        predictionForm.addEventListener('submit', handlePredictionSubmit);
        
        // Add real-time validation
        addFormValidation();
    }
}

/**
 * Add form validation
 */
function addFormValidation() {
    const form = document.getElementById('predictionForm');
    if (!form) return;
    
    const inputs = form.querySelectorAll('input, select');
    
    inputs.forEach(input => {
        input.addEventListener('blur', validateField);
        input.addEventListener('input', clearFieldError);
    });
}

/**
 * Validate individual field
 */
function validateField(event) {
    const field = event.target;
    const value = field.value.trim();
    
    // Remove existing error styling
    clearFieldError(event);
    
    // Check if required field is empty
    if (field.hasAttribute('required') && !value) {
        showFieldError(field, 'This field is required');
        return false;
    }
    
    // Validate specific fields
    if (field.name === 'BMI') {
        const bmi = parseFloat(value);
        if (bmi < 10 || bmi > 100) {
            showFieldError(field, 'BMI must be between 10 and 100');
            return false;
        }
    }
    
    if (field.name === 'PhysicalHealth' || field.name === 'MentalHealth') {
        const days = parseInt(value);
        if (days < 0 || days > 30) {
            showFieldError(field, 'Must be between 0 and 30 days');
            return false;
        }
    }
    
    if (field.name === 'SleepTime') {
        const hours = parseFloat(value);
        if (hours < 0 || hours > 24) {
            showFieldError(field, 'Sleep time must be between 0 and 24 hours');
            return false;
        }
    }
    
    return true;
}

/**
 * Show field error
 */
function showFieldError(field, message) {
    field.classList.add('is-invalid');
    
    // Remove existing error message
    const existingError = field.parentNode.querySelector('.invalid-feedback');
    if (existingError) {
        existingError.remove();
    }
    
    // Add new error message
    const errorDiv = document.createElement('div');
    errorDiv.className = 'invalid-feedback';
    errorDiv.textContent = message;
    field.parentNode.appendChild(errorDiv);
}

/**
 * Clear field error
 */
function clearFieldError(event) {
    const field = event.target;
    field.classList.remove('is-invalid');
    
    const errorMessage = field.parentNode.querySelector('.invalid-feedback');
    if (errorMessage) {
        errorMessage.remove();
    }
}

/**
 * Handle prediction form submission
 */
async function handlePredictionSubmit(event) {
    event.preventDefault();
    
    const form = event.target;
    const submitBtn = document.getElementById('submitBtn');
    const resultsSection = document.getElementById('resultsSection');
    
    // Validate all fields
    const inputs = form.querySelectorAll('input[required], select[required]');
    let isValid = true;
    
    inputs.forEach(input => {
        if (!validateField({ target: input })) {
            isValid = false;
        }
    });
    
    if (!isValid) {
        showNotification('Please fix the errors in the form', 'error');
        return;
    }
    
    // Show loading state
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
    submitBtn.disabled = true;
    
    try {
        // Get form data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData);
        
        // Convert numeric fields
        data.BMI = parseFloat(data.BMI);
        data.PhysicalHealth = parseInt(data.PhysicalHealth);
        data.MentalHealth = parseInt(data.MentalHealth);
        data.SleepTime = parseFloat(data.SleepTime);
        
        // Send prediction request
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            // Display results
            displayPredictionResults(result);
            resultsSection.style.display = 'block';
            resultsSection.scrollIntoView({ behavior: 'smooth' });
            
            // Show success notification
            showNotification('Prediction completed successfully!', 'success');
        } else {
            throw new Error(result.error || 'Prediction failed');
        }
        
    } catch (error) {
        console.error('Prediction error:', error);
        showNotification('Error: ' + error.message, 'error');
    } finally {
        // Reset button
        submitBtn.innerHTML = '<i class="fas fa-calculator me-2"></i>Get Prediction';
        submitBtn.disabled = false;
    }
}

/**
 * Display prediction results
 */
function displayPredictionResults(result) {
    const resultsDiv = document.getElementById('predictionResults');
    
    const riskColor = result.prediction === 1 ? 'danger' : 'success';
    const riskIcon = result.prediction === 1 ? 'fas fa-exclamation-triangle' : 'fas fa-check-circle';
    const riskText = result.prediction === 1 ? 'High Risk' : 'Low Risk';
    const riskPercentage = (result.probability.disease * 100).toFixed(1);
    
    resultsDiv.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <div class="result-card">
                    <h5><i class="${riskIcon} me-2"></i>Risk Assessment</h5>
                    <div class="risk-indicator risk-${riskColor}">
                        <h3>${riskText}</h3>
                        <p>Risk Level: ${result.risk_level}</p>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="result-card">
                    <h5><i class="fas fa-percentage me-2"></i>Probability</h5>
                    <div class="probability-chart">
                        <div class="progress mb-2">
                            <div class="progress-bar bg-${riskColor}" style="width: ${riskPercentage}%">
                                ${riskPercentage}%
                            </div>
                        </div>
                        <p class="mb-0">Heart Disease Risk: ${riskPercentage}%</p>
                        <p class="mb-0 text-muted">No Disease Risk: ${(100 - riskPercentage).toFixed(1)}%</p>
                    </div>
                </div>
            </div>
        </div>
        
        ${result.recommendations && result.recommendations.length > 0 ? `
        <div class="recommendations mt-4">
            <h5><i class="fas fa-lightbulb me-2"></i>Personalized Recommendations</h5>
            <div class="row">
                <div class="col-md-12">
                    <div class="list-group">
                        ${result.recommendations.map(rec => `
                            <div class="list-group-item d-flex align-items-center">
                                <i class="fas fa-arrow-right me-3 text-primary"></i>
                                <span>${rec}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        </div>
        ` : ''}
        
        <div class="mt-4 text-center">
            <button class="btn btn-outline-primary" onclick="window.print()">
                <i class="fas fa-print me-2"></i>Print Results
            </button>
            <button class="btn btn-primary ms-2" onclick="location.reload()">
                <i class="fas fa-redo me-2"></i>New Prediction
            </button>
        </div>
    `;
}

/**
 * Initialize analysis page
 */
async function initializeAnalysis() {
    try {
        // Show loading spinner
        const loadingSpinner = document.getElementById('loadingSpinner');
        const analysisContent = document.getElementById('analysisContent');
        
        if (loadingSpinner) {
            loadingSpinner.style.display = 'block';
        }
        
        // Load analysis data
        const response = await fetch('/api/analysis');
        if (!response.ok) {
            throw new Error('Failed to load analysis data');
        }
        
        analysisData = await response.json();
        
        // Display dataset information
        displayDatasetInfo(analysisData.dataset_info);
        
        // Create charts
        createFeatureImportanceChart(analysisData.feature_importance);
        createRiskFactorsChart(analysisData.risk_factors);
        
        // Hide loading spinner and show content
        if (loadingSpinner) {
            loadingSpinner.style.display = 'none';
        }
        if (analysisContent) {
            analysisContent.style.display = 'block';
        }
        
    } catch (error) {
        console.error('Error loading analysis:', error);
        const loadingSpinner = document.getElementById('loadingSpinner');
        if (loadingSpinner) {
            loadingSpinner.innerHTML = 
                '<div class="alert alert-danger">Error loading analysis data. Please try again later.</div>';
        }
    }
}

/**
 * Display dataset information
 */
function displayDatasetInfo(datasetInfo) {
    const datasetDiv = document.getElementById('datasetInfo');
    if (!datasetDiv) return;
    
    datasetDiv.innerHTML = `
        <div class="col-md-3">
            <div class="stat-card">
                <h3 class="stat-number">${datasetInfo.total_samples.toLocaleString()}</h3>
                <p class="stat-label">Total Samples</p>
            </div>
        </div>
        <div class="col-md-3">
            <div class="stat-card">
                <h3 class="stat-number">${datasetInfo.features}</h3>
                <p class="stat-label">Features</p>
            </div>
        </div>
        <div class="col-md-3">
            <div class="stat-card">
                <h3 class="stat-number">${datasetInfo.target_distribution.Yes || 0}</h3>
                <p class="stat-label">Heart Disease Cases</p>
            </div>
        </div>
        <div class="col-md-3">
            <div class="stat-card">
                <h3 class="stat-number">${datasetInfo.target_distribution.No || 0}</h3>
                <p class="stat-label">Healthy Cases</p>
            </div>
        </div>
    `;
}

/**
 * Create feature importance chart
 */
function createFeatureImportanceChart(featureImportance) {
    const ctx = document.getElementById('featureImportanceChart');
    if (!ctx) return;
    
    const features = Object.keys(featureImportance);
    const values = Object.values(featureImportance);
    
    new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: features,
            datasets: [{
                label: 'Importance',
                data: values,
                backgroundColor: 'rgba(231, 76, 60, 0.8)',
                borderColor: 'rgba(231, 76, 60, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Feature Importance in Heart Disease Prediction',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                },
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Importance Score'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Features'
                    }
                }
            }
        }
    });
}

/**
 * Create risk factors chart
 */
function createRiskFactorsChart(riskFactors) {
    const ctx = document.getElementById('riskFactorsChart');
    if (!ctx) return;
    
    const factors = Object.keys(riskFactors);
    const yesRates = factors.map(factor => riskFactors[factor].Yes);
    const noRates = factors.map(factor => riskFactors[factor].No);
    
    new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: factors,
            datasets: [
                {
                    label: 'With Factor',
                    data: yesRates,
                    backgroundColor: 'rgba(231, 76, 60, 0.8)',
                    borderColor: 'rgba(231, 76, 60, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Without Factor',
                    data: noRates,
                    backgroundColor: 'rgba(52, 152, 219, 0.8)',
                    borderColor: 'rgba(52, 152, 219, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Heart Disease Rate by Risk Factors',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    title: {
                        display: true,
                        text: 'Heart Disease Rate'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Risk Factors'
                    }
                }
            }
        }
    });
}

/**
 * Load health tips
 */
async function loadHealthTips() {
    try {
        const response = await fetch('/api/health-tips');
        const tips = await response.json();
        
        const tipsDiv = document.getElementById('healthTips');
        if (!tipsDiv) return;
        
        tipsDiv.innerHTML = `
            <div class="row">
                <div class="col-md-4">
                    <div class="card h-100">
                        <div class="card-body">
                            <h5 class="card-title"><i class="fas fa-heart me-2 text-primary"></i>General Health</h5>
                            <ul class="list-group list-group-flush">
                                ${tips.general.map(tip => `<li class="list-group-item">${tip}</li>`).join('')}
                            </ul>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card h-100">
                        <div class="card-body">
                            <h5 class="card-title"><i class="fas fa-heartbeat me-2 text-danger"></i>Heart Health</h5>
                            <ul class="list-group list-group-flush">
                                ${tips.heart_health.map(tip => `<li class="list-group-item">${tip}</li>`).join('')}
                            </ul>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card h-100">
                        <div class="card-body">
                            <h5 class="card-title"><i class="fas fa-shield-alt me-2 text-success"></i>Prevention</h5>
                            <ul class="list-group list-group-flush">
                                ${tips.prevention.map(tip => `<li class="list-group-item">${tip}</li>`).join('')}
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        `;
    } catch (error) {
        console.error('Error loading health tips:', error);
    }
}

/**
 * Initialize animations
 */
function initializeAnimations() {
    // Add fade-in animation to elements
    const animatedElements = document.querySelectorAll('.card, .feature-card, .step-card');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in-up');
            }
        });
    }, {
        threshold: 0.1
    });
    
    animatedElements.forEach(el => {
        observer.observe(el);
    });
}

/**
 * Initialize tooltips
 */
function initializeTooltips() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Show notification
 */
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

/**
 * Utility function to format numbers
 */
function formatNumber(num) {
    return new Intl.NumberFormat().format(num);
}

/**
 * Utility function to format percentages
 */
function formatPercentage(num) {
    return (num * 100).toFixed(1) + '%';
}

/**
 * Smooth scroll to element
 */
function smoothScrollTo(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
    }
}

/**
 * Copy text to clipboard
 */
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Copied to clipboard!', 'success');
    }).catch(() => {
        showNotification('Failed to copy to clipboard', 'error');
    });
}

/**
 * Export results as JSON
 */
function exportResults() {
    const results = document.getElementById('predictionResults');
    if (results) {
        const data = {
            timestamp: new Date().toISOString(),
            results: results.innerHTML
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'heart-disease-prediction-results.json';
        a.click();
        URL.revokeObjectURL(url);
    }
}

// Export functions for global access
window.HeartGuardAI = {
    smoothScrollTo,
    copyToClipboard,
    exportResults,
    showNotification
};
