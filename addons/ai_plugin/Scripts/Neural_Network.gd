@tool
class_name NeuralNetwork

var input_nodes: int
var hidden_nodes: int
var output_nodes: int

var weights_input_hidden: Matrix
var weights_hidden_output: Matrix

var bias_hidden: Matrix
var bias_output: Matrix

var learning_rate: float

var activation_function: Callable
var activation_dfunction: Callable

var fitness: float = 0.0

var color: Color = Color.TRANSPARENT

var raycasts: Array


func _init(_input_nodes: int, _hidden_nodes: int, _output_nodes: int) -> void:
	
	assert(_input_nodes > 0, "Input Nodes must be set to a value above 0. (Input nodes > 0)")
	assert(_hidden_nodes > 0, "Hidden Nodes must be set to a value above 0. (Hidden nodes > 0)")
	assert(_output_nodes > 0, "Output Nodes must be set to a value above 0. (Output nodes > 0)")
	
	randomize()
	input_nodes = _input_nodes;
	hidden_nodes = _hidden_nodes;
	output_nodes = _output_nodes;
	
	weights_input_hidden = Matrix.rand(Matrix.new(hidden_nodes, input_nodes))
	weights_hidden_output = Matrix.rand(Matrix.new(output_nodes, hidden_nodes));
	
	bias_hidden = Matrix.rand(Matrix.new(hidden_nodes, 1));
	bias_output = Matrix.rand(Matrix.new(output_nodes, 1));
	
	set_learning_rate()
	set_activation_function()
	set_random_color()

func set_color(_color: Color):
	color = _color

func set_random_color():
	if color is Color.TRANSPARENT:
		color = Color(randi_range(0, 1), randi_range(0, 1), randi_range(0, 1))

func set_learning_rate(_learning_rate: float = 0.1) -> void:
	
	assert(_learning_rate > 0, "Learning Rate must be set higher than 0. (Learning Rate > 0)")
	
	learning_rate = _learning_rate

func set_activation_function(callback: Callable = Callable(Activation, "sigmoid"), dcallback: Callable = Callable(Activation, "dsigmoid")) -> void:
	activation_function = callback
	activation_dfunction = dcallback

func predict(input_array: Array) -> Array:
	
	assert(input_array.size() == input_nodes, "Number of inputs must be equal to the number of input nodes set.")
	
	var inputs = Matrix.from_array(input_array)
	
	var hidden = Matrix.product(weights_input_hidden, inputs)
	hidden = Matrix.add(hidden, bias_hidden)
	hidden = Matrix.map(hidden, activation_function)

	var output = Matrix.product(weights_hidden_output, hidden)
	output = Matrix.add(output, bias_output)
	output = Matrix.map(output, activation_function)

	return Matrix.to_array(output)

func train(input_array: Array, target_array: Array):
	
	assert(input_array.size() == input_nodes, "Number of inputs must be equal to the number of input nodes set.")
	assert(target_array.size() == output_nodes, "Number of outputs / targets must be equal to the number of outputs nodes set.")
	
	var inputs = Matrix.from_array(input_array)
	var targets = Matrix.from_array(target_array)
	
	var hidden = Matrix.product(weights_input_hidden, inputs);
	hidden = Matrix.add(hidden, bias_hidden)
	hidden = Matrix.map(hidden, activation_function)
	
	var outputs = Matrix.product(weights_hidden_output, hidden)
	outputs = Matrix.add(outputs, bias_output)
	outputs = Matrix.map(outputs, activation_function)
	
	var output_errors = Matrix.subtract(targets, outputs)
	
	var gradients = Matrix.map(outputs, activation_dfunction)
	gradients = Matrix.multiply(gradients, output_errors)
	gradients = Matrix.scalar(gradients, learning_rate)
	
	var hidden_t = Matrix.transpose(hidden)
	var weight_ho_deltas = Matrix.product(gradients, hidden_t)
	
	weights_hidden_output = Matrix.add(weights_hidden_output, weight_ho_deltas)
	bias_output = Matrix.add(bias_output, gradients)
	
	var weights_hidden_output_t = Matrix.transpose(weights_hidden_output)
	var hidden_errors = Matrix.product(weights_hidden_output_t, output_errors)
	
	var hidden_gradient = Matrix.map(hidden, activation_dfunction)
	hidden_gradient = Matrix.multiply(hidden_gradient, hidden_errors)
	hidden_gradient = Matrix.scalar(hidden_gradient, learning_rate)
	
	var inputs_t = Matrix.transpose(inputs)
	var weight_ih_deltas = Matrix.product(hidden_gradient, inputs_t)

	weights_input_hidden = Matrix.add(weights_input_hidden, weight_ih_deltas)

	bias_hidden = Matrix.add(bias_hidden, hidden_gradient)

func add_raycast(_raycast: Object):
	assert((_raycast is RayCast2D) or (_raycast is RayCast3D), "Raycast must either be a RayCast2D or RayCast3D.")
	raycasts.append(_raycast)

func get_inputs_from_raycasts() -> Array:
	
	var _input_array: Array[float]
	
	for ray in raycasts:
		_input_array.append(get_distance(ray))
	
	return _input_array

func get_prediction_from_raycasts(optional_val: Array) -> Array:
	var _array_ = get_inputs_from_raycasts()
	_array_.append_array(optional_val)
	return predict(_array_)

func get_distance(_raycast: Object):
	assert((_raycast is RayCast2D) or (_raycast is RayCast3D), "Raycast must either be a RayCast2D or RayCast3D.")
	var distance: float = 0.0
	
	if _raycast.is_colliding():
		var origin = _raycast.global_transform.get_origin()
		var collision = _raycast.get_collision_point()
		distance = origin.distance_to(collision)

	return distance

static func reproduce(a: NeuralNetwork, b: NeuralNetwork) -> NeuralNetwork:
	
	assert(a.input_nodes == b.input_nodes, "The Neural Networks must have equal Input Nodes.")
	assert(a.hidden_nodes == b.hidden_nodes, "The Neural Networks must have equal Hidden Nodes.")
	assert(a.output_nodes == b.output_nodes, "The Neural Networks must have equal Output Nodes.")
	
	var result = NeuralNetwork.new(a.input_nodes, a.hidden_nodes, a.output_nodes)

	result.weights_input_hidden = Matrix.random(a.weights_input_hidden, b.weights_input_hidden)
	result.weights_hidden_output = Matrix.random(a.weights_hidden_output, b.weights_hidden_output)
	result.bias_hidden = Matrix.random(a.bias_hidden, b.bias_hidden)
	result.bias_output = Matrix.random(a.bias_output, b.bias_output)

	return result

static func mutate(nn: NeuralNetwork, mutation_chance: float = 0.3, mutation_range: Vector2 = Vector2(-0.2, 0.2), callback: Callable = Callable(NeuralNetwork, "mutate_callable")) -> NeuralNetwork:
	var result = NeuralNetwork.new(nn.input_nodes, nn.hidden_nodes, nn.output_nodes)
	result.weights_input_hidden = Matrix.mutate_map(nn.weights_input_hidden, callback, mutation_chance, mutation_range)
	result.weights_hidden_output = Matrix.mutate_map(nn.weights_hidden_output, callback, mutation_chance, mutation_range)
	result.bias_hidden = Matrix.mutate_map(nn.bias_hidden, callback, mutation_chance, mutation_range)
	result.bias_output = Matrix.mutate_map(nn.bias_output, callback, mutation_chance, mutation_range)
	return result

static func copy(nn : NeuralNetwork) -> NeuralNetwork:
	
	var result = NeuralNetwork.new(nn.input_nodes, nn.hidden_nodes, nn.output_nodes)
	result.weights_input_hidden = Matrix.copy(nn.weights_input_hidden)
	result.weights_hidden_output = Matrix.copy(nn.weights_hidden_output)
	result.bias_hidden = Matrix.copy(nn.bias_hidden)
	result.bias_output = Matrix.copy(nn.bias_output)
	result.color = nn.color
	
	return result

static func mutate_callable(value, _row, _col, mutation_chance: float, mutation_range: Vector2):
	randomize()
	if randf_range(0, 1) < mutation_chance:
		value += randf_range(mutation_range.x, mutation_range.y)
		
	return value

