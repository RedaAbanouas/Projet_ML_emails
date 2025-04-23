document.getElementById('predict-btn').addEventListener('click', async (event) => {
    event.preventDefault(); 


    const model = document.querySelector('input[name="model"]:checked')?.value;

 
    const email = document.getElementById('email').value;


    if (!model || !email) {
        alert('Please select a model and e-mail.');
        return;
    }


    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `model=${encodeURIComponent(model)}&email=${encodeURIComponent(email)}`,
    });


    const result = await response.json();
    const resultDiv = document.querySelector('.pred-img');
    const body = document.querySelector('body');
    const predres = document.querySelector('.pred-res');
    const predLabel = document.querySelector('.pred-label');

    if (response.ok) {
        if (result.prediction === 'SPAM') {
            body.classList.add('spam-bg');
            body.classList.remove('not-spam-bg');

            predres.classList.add('spam-predres');
            predres.classList.remove('not-spam-predres');

            predLabel.classList.add('spam-pred-label');
            predLabel.classList.remove('not-spam-pred-label');

            resultDiv.innerHTML = `<div class="pred-output spam-output"> <i class="ri-mail-close-fill" style="font-size: 5vw"></i> <span>SPAM</span> </div>`;
        } else if (result.prediction === 'NOT SPAM') {
            body.classList.remove('spam-bg');
            body.classList.add('not-spam-bg');

            predres.classList.remove('spam-predres');
            predres.classList.add('not-spam-predres');

            predLabel.classList.remove('spam-pred-label');
            predLabel.classList.add('not-spam-pred-label');

            resultDiv.innerHTML = `<div class="pred-output not-spam-output"> <i class="ri-mail-check-fill" style="font-size: 5vw"></i> <span>NOT SPAM</span> </div>`;
        } 

        setTimeout(() => {
            const predOutput = document.querySelector('.pred-output');
            if (predOutput) {
                predOutput.classList.add('visible');
            }
        }, 100);


    } else {
        resultDiv.innerHTML = `<div class="pred-output"><span>Error: ${result.error}</span></div>`;
    }
});

document.getElementById('toggle-dark').addEventListener('click', function () {
    document.body.classList.toggle('dark');
    
});
