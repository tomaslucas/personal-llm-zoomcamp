## Homework: Open-Source LLMs

In this homework, we'll experiment more with Ollama

> It's possible that your answers won't match exactly. If it's the case, select the closest one.

## Q1. Running Ollama with Docker

Let's run ollama with Docker. We will need to execute the 
same command as in the lectures:

```bash
docker run -it \
    --rm \
    -v ollama:/root/.ollama \
    -p 11434:11434 \
    --name ollama \
    ollama/ollama
```

What's the version of ollama client? 

To find out, enter the container and execute `ollama` with the `-v` flag.

`answer:` ollama version is 0.1.45


## Q2. Downloading an LLM 

We will donwload a smaller LLM - gemma:2b. 

Again let's enter the container and pull the model:

```bash
ollama pull gemma:2b
```

In docker, it saved the results into `/root/.ollama`

We're interested in the metadata about this model. You can find
it in `models/manifests/registry.ollama.ai/library`

What's the content of the file related to gemma?

`answer:` {"schemaVersion":2,"mediaType":"application/vnd.docker.distribution.manifest.v2+json","config":{"mediaType":"application/vnd.docker.container.image.v1+json","digest":"sha256:887433b89a901c156f7e6944442f3c9e57f3c55d6ed52042cbb7303aea994290","size":483},"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:c1864a5eb19305c40519da12cc543519e48a0697ecd30e15d5ac228644957d12","size":1678447520},{"mediaType":"application/vnd.ollama.image.license","digest":"sha256:097a36493f718248845233af1d3fefe7a303f864fae13bc31a3a9704229378ca","size":8433},{"mediaType":"application/vnd.ollama.image.template","digest":"sha256:109037bec39c0becc8221222ae23557559bc594290945a2c4221ab4f303b8871","size":136},{"mediaType":"application/vnd.ollama.image.params","digest":"sha256:22a838ceb7fb22755a3b0ae9b4eadde629d19be1f651f73efb8c6b4e2cd0eea0","size":84}]}

## Q3. Running the LLM

Test the following prompt: "10 * 10". What's the answer?

`answer:` 
Sure, here's the answer to your question:

10 * 10 = 100.

## Q4. Donwloading the weights 

We don't want to pull the weights every time we run
a docker container. Let's do it once and have them available
every time we start a container.

First, we will need to change how we run the container.

Instead of mapping the `/root/.ollama` folder to a named volume,
let's map it to a local directory:

```bash
mkdir ollama_files

docker run -it \
    --rm \
    -v ./ollama_files:/root/.ollama \
    -p 11434:11434 \
    --name ollama \
    ollama/ollama
```

Now pull the model:

```bash
docker exec -it ollama ollama pull gemma:2b 
```

What's the size of the `ollama_files/models` folder? 

* 0.6G
* 1.2G
* 1.7G
* 2.2G

Hint: on linux, you can use `du -h` for that.

`answer:` 1,6G	ollama_files -> the closer is 1.7G

## Q5. Adding the weights 

Let's now stop the container and add the weights 
to a new image

For that, let's create a `Dockerfile`:

```dockerfile
FROM ollama/ollama

COPY ...
```

- Step 1: Change the ownership of the local directory to my user:

```bash
sudo chown aaaa:aaaa ollama_files
```

- I have errors with the directory, so I have to restart docker service

```bash
sudo systemctl restart docker
```
- Step 3: Use the Dockerfile to build the image from the place where I have the directory

```bash
docker build -t my-ollama . 


```dockerfile
FROM ollama/ollama

COPY ~/ollama_files /root/.ollama
```

What do you put after `COPY`?

`answer:` 2.22GB

## Q6. Serving it 

Let's build it:

```bash
docker build -t ollama-gemma2b .
```

And run it:

```bash
docker run -it --rm -p 11434:11434 ollama-gemma2b
```

We can connect to it using the OpenAI client

Let's test it with the following prompt:

```python
prompt = "What's the formula for energy?"
```

Also, to make results reproducible, set the `temperature` parameter to 0:

```bash
response = client.chat.completions.create(
    #...
    temperature=0.0
)
```

- Steps 1: Client:

```python
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

open_source = True

if open_source:  
    client = OpenAI(
        base_url='http://localhost:11434/v1/',
        api_key='ollama',
    )
else:
    client = OpenAI()
```

- Steps 2: Prompt:

```python
    response = client.chat.completions.create(
        model="gemma:2b",
        temperature=0.0,
        messages= [
            {"role": "user", "content": prompt},
        ]
    )
```
- Steps 3: Show response and get completion_tokens

```
response

ChatCompletion(id='chatcmpl-570', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content="Sure, here's the formula for energy:\n\n**E = K + U**\n\nWhere:\n\n* **E** is the energy in joules (J)\n* **K** is the kinetic energy in joules (J)\n* **U** is the potential energy in joules (J)\n\n**Kinetic energy (K)** is the energy an object possesses when it moves or is in motion. It is calculated as half the product of its mass (m) and its velocity (v) squared:\n\n**K = 1/2 * m * v^2**\n\n**Potential energy (U)** is the energy an object possesses due to its position or configuration. It is calculated as the product of its mass and the gravitational constant (g) multiplied by the height or distance of its position.\n\n**Gravitational potential energy (U)** is given by the formula:\n\n**U = mgh**\n\nWhere:\n\n* **m** is the mass of the object in kilograms (kg)\n* **g** is the acceleration due to gravity in meters per second squared (m/s^2)\n* **h** is the height or distance of the object in meters (m)\n\nThe formula for energy tells us that energy can be transferred or converted from one form to another. For example, when a ball is thrown, its kinetic energy is converted into potential energy as it rises. When a car brakes, its potential energy is converted into kinetic energy.", role='assistant', function_call=None, tool_calls=None))], created=1719942062, model='gemma:2b', object='chat.completion', system_fingerprint='fp_ollama', usage=CompletionUsage(completion_tokens=306, prompt_tokens=34, total_tokens=340))
```


How many completion tokens did you get in response?

* 304
* 604
* 904
* 1204

## Submit the results

* Submit your results here: https://courses.datatalks.club/llm-zoomcamp-2024/homework/hw2
* It's possible that your answers won't match exactly. If it's the case, select the closest one.
