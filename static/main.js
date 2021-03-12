function myFunction() {
  var checkBox = document.getElementById("showTextBox");
  var queryText = document.getElementById("query-text");
  var queryUrl = document.getElementById("query-url");
  if (checkBox.checked == true){
    queryText.style.display = "block";
	queryUrl.style.display = "none";
  } else {
	 queryUrl.style.display = "block";
     queryText.style.display = "none";
  }
}

function myFunction2() {
  var checkBox = document.getElementById("eval_select");
  var evaluation = document.getElementById("evaluation");
  
  if (checkBox.checked == true){
    evaluation.style.display = "block";
	
  } else {
	
     evaluation.style.display = "none";
  }
}