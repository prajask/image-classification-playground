/* --- Send Image Classification Request --- */

var request = new XMLHttpRequest();
request.open("GET", classification_url, true);
request.onload = function(){
    if(request.response == "Done"){
        window.location = "/classified_images";
    }
}
request.send();