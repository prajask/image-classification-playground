/* --- Image Upload Form Validator --- */

const file_input_form = document.querySelector("#file-input-form");
const file_input = document.querySelector("#image-file-input");
const input_label = document.querySelector("#input-label-text");
var files_selected = false;

//-- Change Label According to Number of Images Selected 
file_input.onchange = function(){
    input_label.innerText = `${file_input.files.length} Files Selected`;
    
    if(file_input.files.length > 0){
        files_selected = true;
    }

    else{
        files_selected = false;
    }

}

function validate(){
    if(files_selected){
        file_input_form.submit();
    }

    else{
        input_label.classList.add("error");
        input_label.innerText = "Select Files to Continue...";
    }
}