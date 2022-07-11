/* Model Layers Manipulation Functions */

//-- Input Fields
const number_of_layers_before_flatten = document.querySelector("#number_of_layers_before_flatten");
const number_of_layers_after_flatten = document.querySelector("#number_of_layers_after_flatten");

const layers = document.querySelector("#model-layers");
const flatten_layer = document.querySelector("#flatten_layer");

const add_layer_before_flatten_button = document.querySelector("#add_layer_before_flatten");
add_layer_before_flatten_button.onclick = add_layer_before_flatten;

const add_layer_after_flatten_button = document.querySelector("#add_layer_after_flatten");
add_layer_after_flatten_button.onclick = add_layer_after_flatten;

//-- Insert Layers Before the Flatten Layer
function add_layer_before_flatten(){
    const layer_number = parseInt(number_of_layers_before_flatten.value) + parseInt(number_of_layers_after_flatten.value) + 4;

    const layer = document.createElement("div");
    layer.classList.add("card");
    const layer_id = `new_layer_${layer_number}`;
    layer.setAttribute("id", layer_id);

    const inner_html = `<p class="title">New Layer ${layer_number}</p>
        <label for="${layer_id}_class">Class</label>
        <select id="${layer_id}_class" name="${layer_id}_class" onchange="change_layer_class(this)">
            <option value="Conv2D" selected>Convolution</option>
            <option value="MaxPool2D">Max Pooling</option>
            <option value="Dropout">Dropout</option>
        </select>

        <div id="${layer_id}_parameters">
            ${get_layer_parameters(layer_id, "Conv2D")}
        </div>

        <button type="button" class="button remove-layer" onclick="remove_layer_before_flatten(this)">Remove Layer</button>
    `;

    layer.innerHTML = inner_html;

    layers.insertBefore(layer, add_layer_before_flatten_button);

    number_of_layers_before_flatten.value = parseInt(number_of_layers_before_flatten.value) + 1;
}

//-- Insert Layers After the Flatten Layer
function add_layer_after_flatten(){
    const layer_number = parseInt(number_of_layers_before_flatten.value) + parseInt(number_of_layers_after_flatten.value) + 4;

    const layer = document.createElement("div");
    layer.classList.add("card");
    const layer_id = `new_layer_${layer_number}`;
    layer.setAttribute("id", layer_id);

    const inner_html = `<p class="title">New Layer ${layer_number}</p>
        <label for="${layer_id}_class">Class</label>
        <select id="${layer_id}_class" name="${layer_id}_class" onchange="change_layer_class(this)">
            <option value="Dense" selected>Dense</option>
            <option value="Dropout">Dropout</option>
        </select>

        <div id=${layer_id}_parameters>
            ${get_layer_parameters(layer_id, "Dense")}
        </div>

        <button type="button" class="button remove-layer" onclick="remove_layer_after_flatten(this)">Remove Layer</button>
    `;

    layer.innerHTML = inner_html;

    layers.insertBefore(layer, add_layer_after_flatten_button.nextSibling);

    number_of_layers_after_flatten.value = parseInt(number_of_layers_after_flatten.value) + 1;
}

//-- Return Layer Parameters According to Layer Class Selected 
function get_layer_parameters(layer_id, layer_class){
    switch(layer_class){
        case "Conv2D":
            return `<label for="${layer_id}_filter_size">Filter Size</label>
                <input type="text" class="filter_size" id="${layer_id}_filter_size" name="${layer_id}_filter_size" value="150" required/>
            
                <label for="${layer_id}_kernel_size">Kernel Size</label>
                <input type="text" class="kernel_size" id="${layer_id}_kernel_size" name="${layer_id}_kernel_size" value="3" required/>
            
                <label for="${layer_id}_activation">Activation Function</label>
                <select id="${layer_id}_activation" name="${layer_id}_activation">
                    <option value="relu" selected>relu</option>
                    <option value="elu">elu</option>
                    <option value="selu">selu</option>
                    <option value="softmax">softmax</option>
                    <option value="softplus">softplus</option>
                    <option value="softsign">softsign</option>
                    <option value="tanh">tanh</option>
                    <option value="sigmoid">sigmoid</option>
                    <option value="hard_sigmoid">hard_sigmoid</option>
                    <option value="exponential">exponential</option>
                    <option value="linear">linear</option>
                </select>
            `;

        case "MaxPool2D":
            return `<label for="${layer_id}_pool_size">Pool Size</label>
            <input type="text" class="pool_size" id="${layer_id}_pool_size" name="${layer_id}_pool_size" value="5" required>
            `;

        case "Dropout":
            return `<label for="${layer_id}_rate">Drop Rate</label>
            <input type="text" class="drop_rate" id="${layer_id}_rate" name="${layer_id}_rate" value="0.5" required>
            `;

        case "Dense":
            return `<label for="${layer_id}_units">Units</label>
                <input type="text" class="units" id="${layer_id}_units" name="${layer_id}_units" value="150" required/>

                <label for="${layer_id}_activation">Activation Function</label>
                <select id="${layer_id}_activation" name="${layer_id}_activation">
                    <option value="relu" selected>relu</option>
                    <option value="elu">elu</option>
                    <option value="selu">selu</option>
                    <option value="softmax">softmax</option>
                    <option value="softplus">softplus</option>
                    <option value="softsign">softsign</option>
                    <option value="tanh">tanh</option>
                    <option value="sigmoid">sigmoid</option>
                    <option value="hard_sigmoid">hard_sigmoid</option>
                    <option value="exponential">exponential</option>
                    <option value="linear">linear</option>
                </select>
            `;
    }
}

//-- Remove Layers Before the Flatten Layer
function remove_layer_before_flatten(layer){
    layer.parentElement.remove();
    number_of_layers_before_flatten.value = parseInt(number_of_layers_before_flatten.value) - 1;
}

//-- Remove Layers After the Flatten Layer
function remove_layer_after_flatten(layer){
    layer.parentElement.remove();
    number_of_layers_after_flatten.value = parseInt(number_of_layers_after_flatten.value) - 1;
}

//-- Edit Layer Parameters if Layer Class is Changed
function change_layer_class(layer_class_input){
    
    const layer = layer_class_input.parentElement;
    const layer_parameters = document.querySelector(`#${layer.id}_parameters`);

    layer_parameters.innerHTML = get_layer_parameters(layer.id, layer_class_input.value);
}

//-- Validate Layer Parameters
function validate_layers(){
    valid = true;
    red = "#FF333A";

    validate_model_name();

    validate_filter_size_fields();

    validate_kernel_size_fields();

    validate_pool_size_fields();

    validate_drop_rate_fields();

    validate_dense_units_fields();

    validate_learning_rate();

    if(valid){
        document.querySelector("#tune-model-form").submit();
    }
}

//--- Validate Model Name Field
function validate_model_name(){
    const model_name_validator = /^[a-zA-Z(\s)*]{2,}$/;
    const model_name_field = document.querySelector("#name");

    if( !model_name_validator.test(model_name_field.value) ){
        model_name_field.style.border = `1px solid ${red}`;
        model_name_field.focus();
        valid = false;
    }

    else{
        model_name_field.style.border = "1px solid black";
    }
}

//--- Validate All Filter Size Fields
function validate_filter_size_fields(){
    const filter_size_validator = /^[0-9]{1,3}$/;
    const filter_size_fields = document.querySelectorAll(".filter_size");

    filter_size_fields.forEach(filter_size_field => {
        if( !filter_size_validator.test(filter_size_field.value) ){
            filter_size_field.style.border = `1px solid ${red}`;
            filter_size_field.focus();
            valid = false;
        }

        else{
            filter_size_field.style.border = "1px solid black";
        }
    });
}

//--- Validate All Kernel Size Fields
function validate_kernel_size_fields(){
    const kernel_size_validator = /^[0-9]{1,3}$/;
    const kernel_size_fields = document.querySelectorAll(".kernel-size");

    kernel_size_fields.forEach(kernel_size_field => {
        if( !kernel_size_validator.test(kernel_size_field.value) ){
            kernel_size_field.style.border = `1px solid ${red}`;
            kernel_size_field.focus();
            valid = false;
        }

        else{
            kernel_size_field.style.border = "1px solid black";
        }
    });
}

//--- Validate All Pool Size Fields
function validate_pool_size_fields(){
    const pool_size_validator = /^[0-9]{1,2}$/;
    const pool_size_fields = document.querySelectorAll(".pool_size");

    pool_size_fields.forEach(pool_size_field => {
        if( !pool_size_validator.test(pool_size_field.value) ){
            pool_size_field.style.border = `1px solid ${red}`;
            pool_size_field.focus();
            valid = false;
        }

        else{
            pool_size_field.style.border = "1px solid black";
        }
    });
}

//--- Validate All Drop Rate Fields
function validate_drop_rate_fields(){
    const drop_rate_validator = /((0(\.[0-9]*)?)|(1(\.0)?))/;
    const drop_rate_fields = document.querySelectorAll(".drop_rate");

    drop_rate_fields.forEach(drop_rate_field => {
        if( !drop_rate_validator.test(drop_rate_field.value) ){
            drop_rate_field.style.border = `1px solid ${red}`;
            drop_rate_field.focus();
            valid = false;
        }

        else{
            drop_rate_field.style.border = "1px solid black";
        }
    });
}

//--- Validate All Dense Layer Unit Fields
function validate_dense_units_fields(){
    const units_validator = /^[0-9]{1,3}$/;
    const units_fields = document.querySelectorAll(".units");

    units_fields.forEach(units_field => {
        if( !units_validator.test(units_field.value)){
            units_field.style.border = `1px solid ${red}`;
            units_field.focus();
            valid = false;
        }

        else{
            units_field.style.border = "1px solid black";
        }
    });
}

//--- Validate Learning Rate Field
function validate_learning_rate(){
    const learning_rate_validator = /((0(\.[0-9]*)?)|(1(\.0)?))/;
    const learning_rate_field = document.querySelector("#learning_rate");

    if( !learning_rate_validator.test(learning_rate_field.value) ){
        learning_rate_field.style.border = `1px solid ${red}`;
        learning_rate_field.focus();
        valid = false;
    }

    else{
        learning_rate_field.style.border = "1px solid black";
    }
}