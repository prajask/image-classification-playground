/* --- Send Save Model Request --- */

var request = new XMLHttpRequest();
request.open("GET", "/save_model", true);
request.onload = function(){
    if(request.response == "Done"){
        window.location = "/user_models";
    }
}
request.send();