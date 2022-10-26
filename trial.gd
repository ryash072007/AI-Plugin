extends Node2D

var nn: NeuralNetwork = NeuralNetwork.new(2,4,1)

var input_arrays: Array[Array] = [
	[0, 0],
	[0, 1],
	[1, 0],
	[1,1]
]

var output_arrays: Array[Array] = [
	[0],
	[1],
	[1],
	[0]
]

var total_correct_prediction: int = 0
var generations: int = 0

func _ready():
	nn.set_learning_rate(5)

func _physics_process(_delta):
	randomize()
	var selected_idx: int = randi_range(0, 3)
	nn.train(input_arrays[selected_idx], output_arrays[selected_idx])
	generations += 1
	$VBoxContainer/HBoxContainer/gen.text = str(generations)
	$"VBoxContainer/HBoxContainer2/00".text = str(snapped(nn.predict([0, 0])[0], 0.1))
	$"VBoxContainer/HBoxContainer3/01".text = str(snapped(nn.predict([0, 1])[0], 0.1))
	$"VBoxContainer/HBoxContainer4/10".text = str(snapped(nn.predict([1, 0])[0], 0.1))
	$"VBoxContainer/HBoxContainer5/11".text = str(snapped(nn.predict([1, 1])[0], 0.1))
