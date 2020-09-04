const red = "#FF333A";

function validate_email(){
    if( !email_validator.test(email_field.value) ){
        email_field.style.border = `1px solid ${red}`;
        invalid_email_error.style.display = "block";
    }

    else{
        email_form.submit();
    }
}

function validate_password_fields(){
    if( !password_validator.test(password_field.value) ){
        password_field.style.border = `1px solid ${red}`;
    }

    else if(confirm_password_field.value != password_field.value){
        confirm_password_field.style.border = `1px solid ${red}`;
        confirm_password_error.style.display = "block";
    }

    else{
        password_form.submit();
    }
}

function validate_name_fields(){
    if( !name_validator.test(first_name_field.value) ){
        first_name_field.style.border = `1px solid ${red}`;
        first_name_error.style.display = "block";
        last_name_error.style.display = "none";
    }

    else if( !name_validator.test(last_name_field.value) ){
        last_name_field.style.border = `1px solid ${red}`;
        first_name_error.style.display = "none";
        last_name_error.style.display = "block";
    }

    else{
        name_form.submit();
    }
}