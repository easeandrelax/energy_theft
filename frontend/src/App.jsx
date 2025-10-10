import React, { useState } from 'react';

// Configuration
//const API_URL = 'http://127.0.0.1:5000/predict';

const API_URL = 'https://my-energy-api.onrender.com/predict'; // Example public URL
// Define the structure and initial state for the 6 features
const initialFeatures = {
    cons_mean: 250.5,
    cons_total: 75150.0,
    diff_std: 5.12,
    lag1_corr: 0.95,
    month_std: 1.5,
    cons_total_zscore: 0.55,
};

// Define the display labels for the features
const FEATURE_LABELS = {
    cons_mean: '1. Average Consumption',
    cons_total: '2. Total Consumption',
    diff_std: '3. Daily Change StDev',
    lag1_corr: '4. Lag-1 Correlation',
    month_std: '5. Monthly StDev',
    cons_total_zscore: '6. Total Consumption Z-Score',
};

// Main App Component
const App = () => {
    const [features, setFeatures] = useState(initialFeatures);
    const [predictionResult, setPredictionResult] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    // Handler for updating any feature input field
    const handleFeatureChange = (e) => {
        const { id, value } = e.target;
        // Ensure the value is converted to a number/float for state
        setFeatures(prev => ({
            ...prev,
            [id]: parseFloat(value) || 0, // Fallback to 0 if parsing fails
        }));
    };

    const handlePrediction = async () => {
        // 1. Reset state and start loading
        setIsLoading(true);
        setError(null);
        setPredictionResult(null);

        // 2. Validate input and prepare payload
        const payload = {};
        let isValid = true;
        
        for (const [key, value] of Object.entries(features)) {
            if (isNaN(value) || value === null) {
                isValid = false;
                break;
            }
            payload[key] = value;
        }
        
        if (!isValid) {
            setError("Please ensure all 6 input fields contain valid numbers.");
            setIsLoading(false);
            return;
        }

        try {
            // 3. Send POST request with the correctly structured JSON payload
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (!response.ok) {
                // Handle API-level errors (like 400 Bad Request)
                const errorMessage = data.error || `Server returned status ${response.status}`;
                setError(`Prediction Failed: ${errorMessage}`);
                return;
            }

            // 4. Update result state
            setPredictionResult(data);

        } catch (err) {
            // Handle network errors (e.g., Flask server is down)
            setError(`Network Error: Could not reach the API at ${API_URL}. Is your Flask app running?`);
        } finally {
            setIsLoading(false);
        }
    };

    // Determine result colors and messages for UI
    const isTheft = predictionResult?.is_theft;
    const resultColor = isTheft ? 'text-red-600 border-red-400' : 'text-green-600 border-green-400';
    const resultText = isTheft ? '🚨 HIGH RISK OF THEFT' : '✅ LOW RISK (Normal)';
    const probabilityScore = predictionResult ? (predictionResult.probability * 100).toFixed(2) : 'N/A';
    
    return (
        <div className="min-h-screen flex items-center justify-center p-4">
            <div className="w-full max-w-lg bg-white p-8 rounded-xl shadow-2xl space-y-8">

                {/* Header */}
                <h1 className="text-4xl font-extrabold text-gray-900 flex items-center">
                    {/* UPDATED: Added inline style to enforce size and fix the "too big" issue. */}
                    <svg 
                        className="w-8 h-8 mr-3 text-yellow-500" 
                        fill="currentColor" 
                        viewBox="0 0 20 20" 
                        xmlns="http://www.w3.org/2000/svg"
                        style={{ width: '2rem', height: '2rem' }} // Enforce size
                    >
                        <path fillRule="evenodd" d="M11.3 1.046A1 1 0 0112 2v2.102a.853.853 0 001.386.581L16 6.882V18a1 1 0 01-2 0V7.22l-2.614-1.2A.853.853 0 0010 6v3.118a1 1 0 01-2 0V2a1 1 0 011.3-.954z" clipRule="evenodd"></path>
                    </svg>
                    Energy Theft Detection
                </h1>
                <p className="text-gray-600">
                    Enter the 6 aggregated customer consumption features below to predict the likelihood of energy theft.
                </p>

                {/* Feature Input Form */}
                <div id="input-container" className="space-y-4">
                    
                    {Object.keys(initialFeatures).map((key) => (
                        <div key={key} className="input-group flex items-center space-x-4">
                            <label htmlFor={key} className="text-gray-700 font-medium w-[150px]">{FEATURE_LABELS[key]}:</label>
                            <input
                                type="number"
                                id={key}
                                value={features[key]}
                                onChange={handleFeatureChange}
                                className="flex-grow p-2 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500"
                                required
                            />
                        </div>
                    ))}
                    
                    <button 
                        onClick={handlePrediction} 
                        disabled={isLoading}
                        className="w-full py-3 mt-4 bg-blue-600 text-white font-bold rounded-lg hover:bg-blue-700 transition duration-150 shadow-md disabled:bg-blue-400"
                    >
                        {isLoading ? 'Predicting...' : 'Predict Theft Likelihood'}
                    </button>
                </div>

                {/* Result Display */}
                {(predictionResult || error) && (
                    <div 
                        className={`mt-6 p-4 border rounded-lg ${error ? 'border-red-400' : ''}`}
                        style={!error && predictionResult ? { borderColor: isTheft ? '#f87171' : '#4ade80' } : {}}
                    >
                        {error && (
                            <p id="api-error" className="text-sm text-red-500 font-medium">{error}</p>
                        )}
                        
                        {predictionResult && !error && (
                            <>
                                <p id="prediction-status" className={`font-bold text-lg ${resultColor}`}>{resultText}</p>
                                <p id="probability-score" className="text-sm text-gray-600 mt-1">
                                    Theft Probability: {probabilityScore}%
                                </p>
                            </>
                        )}
                    </div>
                )}
                
            </div>
        </div>
    );
};

export default App;

