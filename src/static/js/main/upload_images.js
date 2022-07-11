/* --- Image Upload Form Validator --- */

const file_input_form = document.querySelector("#file-input-form");
const file_input = document.querySelector("#image-file-input");
const input_label = document.querySelector("#input-label-text");
var files_selected;
var valid_dimensions;

//-- Change Label According to Number of Images Selected 
file_input.onchange = function () {
    input_label.innerText = `${file_input.files.length} Files Selected`;

    if (file_input.files.length > 0) {
        files_selected = true;
        valid_dimensions = true;

        var reader = new FileReader();
        var image;

        for (const uploadedImage of file_input.files) {
            reader.readAsDataURL(uploadedImage);
            reader.onload = e => {
                image = new Image();
                image.src = e.target.result;
                image.onload = () => {
                    if (image.width !== 150 || image.height !== 150) {
                        valid_dimensions = false;
                    }
                };
            };
        }
    }

    else {
        files_selected = false;
    }

}

function validate() {
    if (files_selected) {
        if (!valid_dimensions) {
            input_label.classList.add("error");
            input_label.innerText = "Invalid Image Dimesions...";
        }

        else file_input_form.submit();
    }

    else {
        input_label.classList.add("error");
        input_label.innerText = "Select Files to Continue...";
    }
}