document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('csvFileInput');
    const resultDiv = document.getElementById('result');
    const customerIDInput = document.getElementById('customerID');
    let customerData = {};

    fileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        const reader = new FileReader();

        reader.onload = (e) => {
            const csvText = e.target.result;
            parseCSV(csvText);
        };

        reader.readAsText(file);
    });

    function parseCSV(csvText) {
        // Use faster parsing method
        const rows = csvText.split('\n').slice(1); // Skip header
        
        // Use Map for faster lookup
        customerData = new Map(
            rows.map(row => {
                const [customerID, defaultProb, decision] = row.split(',');
                return [
                    customerID, 
                    {
                        decision: parseInt(decision),
                        default_prob: parseFloat(defaultProb)
                    }
                ];
            })
        );
    }

    window.checkCreditworthiness = () => {
        const customerID = customerIDInput.value;
        
        if (!customerData) {
            resultDiv.innerHTML = 'Please upload CSV first';
            return;
        }

        const customer = customerData.get(customerID);

        if (customer) {
            if (customer.decision === 1) {
                resultDiv.innerHTML = `
                    <div class="credit-worthy">
                        Customer is Credit Worthy<br>
                        Default Probability: ${(customer.default_prob * 100).toFixed(2)}%
                    </div>
                `;
            } else {
                resultDiv.innerHTML = `
                    <div class="not-credit-worthy">
                        Customer is Not Credit Worthy<br>
                        Default Probability: ${(customer.default_prob * 100).toFixed(2)}%
                    </div>
                `;
            }
        } else {
            resultDiv.innerHTML = 'Customer ID Not Found';
        }
    };
});
