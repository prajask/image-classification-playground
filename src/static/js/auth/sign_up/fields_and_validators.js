const email_form = document.querySelector("#email-form");
const email_validator = /^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$/;
const email_field = document.querySelector("#email");
const invalid_email_error = document.querySelector("#invalid-email");

const password_form = document.querySelector("#password-form");
const password_field = document.querySelector("#password");
const confirm_password_field = document.querySelector("#confirm_password");
const confirm_password_error = document.querySelector("#confirm-password-error");
const password_validator = /^(?=.*\d)(?=.*[a-z])(?=.*[A-Z])(?!.* )(?=.*[^a-zA-Z0-9]).{8,30}$/;

const name_form = document.querySelector("#name-form");
const name_validator = /^[a-zA-Z(\s)*]{2,}$/;
const first_name_field = document.querySelector("#first_name");
const first_name_error = document.querySelector("#first-name-error");
const last_name_field = document.querySelector("#last_name");
const last_name_error = document.querySelector("#last-name-error");