/* --- Send Train Model Request --- */

var request = new XMLHttpRequest();
request.open("GET", "/train_model", true);
request.onload = function(){
    if(request.response == "Done"){
        window.location = "/trained_model";
    }
}
request.send();