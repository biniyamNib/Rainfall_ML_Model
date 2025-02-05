document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const submitBtn = document.getElementById('submitBtn');
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');
    
    // Collect form data
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());
    
    // Convert string values to numbers where needed
    const numericFields = [
        'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
        'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
        'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
        'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm'
    ];

    numericFields.forEach(field => {
        data[field] = parseFloat(data[field]);
    });

    try {
        submitBtn.disabled = true;
        loading.style.display = 'block';
        result.style.display = 'none';

        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const predictionData = await response.json();
        
        document.getElementById('predictionResult').textContent = predictionData.prediction;
        document.getElementById('probabilityResult').textContent = 
            `${(predictionData.probability * 100).toFixed(1)}%`;
        
        result.style.display = 'block';
    } catch (error) {
        alert('Error making prediction. Please try again.');
        console.error('Error:', error);
    } finally {
        submitBtn.disabled = false;
        loading.style.display = 'none';
    }
});