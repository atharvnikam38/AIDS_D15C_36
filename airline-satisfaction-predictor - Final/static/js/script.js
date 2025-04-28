document.addEventListener('DOMContentLoaded', function() {
    // Tab functionality
    window.openTab = function(tabName) {
        const tabContents = document.getElementsByClassName('tab-content');
        for (let i = 0; i < tabContents.length; i++) {
            tabContents[i].classList.remove('active');
        }
        
        const tabButtons = document.getElementsByClassName('tab-btn');
        for (let i = 0; i < tabButtons.length; i++) {
            tabButtons[i].classList.remove('active');
        }
        
        document.getElementById(tabName).classList.add('active');
        
        // Find the button that triggered this and add active class
        const buttons = document.querySelectorAll('.tab-btn');
        for (let i = 0; i < buttons.length; i++) {
            if (buttons[i].textContent.toLowerCase().includes(tabName.toLowerCase())) {
                buttons[i].classList.add('active');
            }
        }
    }
    
    // File upload functionality
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const fileName = document.getElementById('fileName');
    const batchSubmit = document.getElementById('batchSubmit');
    
    if (uploadArea && fileInput) {
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#3498db';
            uploadArea.style.backgroundColor = 'rgba(52, 152, 219, 0.1)';
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.borderColor = '#bdc3c7';
            uploadArea.style.backgroundColor = 'transparent';
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#bdc3c7';
            uploadArea.style.backgroundColor = 'transparent';
            
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                handleFileSelect();
            }
        });
        
        fileInput.addEventListener('change', handleFileSelect);
    }
    
    function handleFileSelect() {
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            fileName.textContent = `Selected file: ${file.name}`;
            fileName.style.color = '#2ecc71';
            if (batchSubmit) {
                batchSubmit.disabled = false;
            }
        } else {
            fileName.textContent = '';
            if (batchSubmit) {
                batchSubmit.disabled = true;
            }
        }
    }
    
    // Form validation
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            let isValid = true;
            const inputs = this.querySelectorAll('input[required], select[required]');
            
            inputs.forEach(input => {
                if (!input.value) {
                    input.style.borderColor = '#e74c3c';
                    isValid = false;
                } else {
                    input.style.borderColor = '#bdc3c7';
                }
            });
            
            if (!isValid) {
                e.preventDefault();
                alert('Please fill in all required fields.');
            }
        });
    }
    
    // Rating input validation
    const ratingInputs = document.querySelectorAll('input[type="number"][min="1"][max="5"]');
    if (ratingInputs) {
        ratingInputs.forEach(input => {
            input.addEventListener('change', function() {
                if (this.value < 1) this.value = 1;
                if (this.value > 5) this.value = 5;
            });
        });
    }
    
    // Age input validation
    const ageInput = document.getElementById('Age');
    if (ageInput) {
        ageInput.addEventListener('change', function() {
            if (this.value < 1) this.value = 1;
            if (this.value > 120) this.value = 120;
        });
    }
    
    // Delay inputs validation
    const delayInputs = document.querySelectorAll('input[name*="Delay"]');
    if (delayInputs) {
        delayInputs.forEach(input => {
            input.addEventListener('change', function() {
                if (this.value < 0) this.value = 0;
            });
        });
    }
    
    // Flight distance validation
    const flightDistanceInput = document.getElementById('Flight Distance');
    if (flightDistanceInput) {
        flightDistanceInput.addEventListener('change', function() {
            if (this.value < 0) this.value = 0;
        });
    }
  });