const { Ollama } = require( "ollama" )

// Simulates an API call to get flight times
// In a real application, this would fetch data from a live database or API
function get_flight_times ( departure, arrival )
{
	const flights = {
		"LAX-NYC": { departure: "02:00 PM", arrival: "10:30 PM", duration: "8h 30m" },
		"LHR-JFK": { departure: "10:00 AM", arrival: "01:00 PM", duration: "3h 00m" },
		"NYC-LAX": { departure: "08:00 AM", arrival: "11:30 AM", duration: "3h 30m" },
		"JFK-LHR": { departure: "09:00 PM", arrival: "09:00 AM", duration: "12h 00m" },
		"CDG-DXB": { departure: "11:00 AM", arrival: "08:00 PM", duration: "9h 00m" },
		"DXB-CDG": { departure: "03:00 AM", arrival: "07:30 AM", duration: "4h 30m" }
	};

	const key = `${departure}-${arrival}`.toUpperCase();
	return JSON.stringify( flights[key] || { error: "Flight not found" });
}

const get_flight_times_schema = {
	type: "function",
	function: {
		name: "get_flight_times",
		description: "Get the flight times between two cities",
		parameters: {
			type: "object",
			properties: {
				departure: {
					type: "string",
					description: "The departure city (airport code)",
				},
				arrival: {
					type: "string",
					description: "The arrival city (airport code)",
				},
			},
			required: ["departure", "arrival"],
		},
	}
}

const availableFunctions = {
	get_flight_times,
};

function handleToolCalls ( toolCalls, messages )
{
	for ( const tool of toolCalls )
	{
		const functionToCall = availableFunctions[tool.function.name];
		if ( functionToCall )
		{
			const functionResponse = functionToCall(
				tool.function.arguments.departure,
				tool.function.arguments.arrival
			);
			// Add function response to the conversation
			messages.push({
				role: "tool",
				content: functionResponse,
			});
		}
	}
}

void async function main ()
{
	const ollama = new Ollama({ host: "http://127.0.0.1:11434" })
	const model = "llama3.1:8b";

	let messages = [{ role: "user", content: "What is the flight time from New York (NYC) to Los Angeles (LAX)?" }];
	// First API call: Send the query and function description to the model
	const response = await ollama.chat({
		model,
		messages,
		tools: [get_flight_times_schema],
	})
	// Add the model's response to the conversation history
	messages.push( response.message );
	if ( response.message.tool_calls && response.message.tool_calls.length > 0 )
	{
		handleToolCalls( response.message.tool_calls, messages );
	}
	else
	{
		console.log( response.message.content );
		return
	}

	// Second API call: Get second response from the model
	const response2 = await ollama.chat({
		model,
		messages,
	});
	messages.push( response2.message );
	console.log( response2.message.content );

	messages.push({ role: "user", content: "What is the flight time from CDG to DXB?" });
	const response3 = await ollama.chat({
		model,
		messages,
		tools: [get_flight_times_schema]
	});
	messages.push( response3.message );
	if ( response3.message.tool_calls && response3.message.tool_calls.length > 0 )
	{
		handleToolCalls( response3.message.tool_calls, messages );
	}
	else
	{
		console.log( response.message.content );
		return
	}

	const response4 = await ollama.chat({
		model,
		messages,
		tools: [get_flight_times_schema]
	});
	console.log( response4.message.content );
}()