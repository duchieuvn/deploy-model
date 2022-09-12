console.log('start 3')

var loadFile = function(event) {
	var image = document.getElementById('display_image');
	image.src = URL.createObjectURL(event.target.files[0]);
};