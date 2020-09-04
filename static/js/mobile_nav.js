/* --- Mobile Navigation Toggle --- */

const sidebar_toggle = document.querySelector("#sidebar-toggle");
const sidebar = document.querySelector(".sidebar");

var sidebar_on = false;

sidebar_toggle.onclick = function(){
    if(sidebar_on){
        sidebar.style.transform = "translateX(-500px)";
        sidebar_on = !sidebar_on;
    }

    else{
        sidebar.style.transform = "translateX(0px)";
        sidebar_on = !sidebar_on;  
    }
}