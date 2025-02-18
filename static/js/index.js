// Event listener for the form submission
document.getElementById('loginForm').addEventListener('submit', function(event) {
    event.preventDefault();  // Prevent the form from submitting the traditional way

    // Get the form data
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    // Send the data via AJAX to the login endpoint
    fetch('/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email: email, password: password })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // On success, close the modal and redirect or update UI
            alert('Login successful!');
            window.location.href = '/';  // Redirect to home page
        } else {
            // On failure, show the error message
            alert(data.message || 'Login failed');
        }
    })
    .catch(error => {
        alert('An error occurred');
    });
});