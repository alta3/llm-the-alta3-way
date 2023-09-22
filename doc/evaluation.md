# Evaluation 101

## Objectives 
 - Test our various models using HellSwag test prompts


## Procedures

1. HellaSwag is a popular test for LLM's.  The conceit for HellaSwag is "Can a Machine Really Finish Your Sentence?" The trick is the sentences involves very tricky context.

0. Now that we can prompt our Falcon 7b model from `llm/git/lit-gpt`, our Llama 70B from WebUI, and Bard from the browser we can see for ourselves how various models hold up to some Hellswag questions.

    > Note: The Falcon7B can be prompted by entering the following on the command line:

    ```
    python generate/base.py \
    --prompt "YOUR PROMPT GOES HERE" \
    --checkpoint_dir checkpoints/tiiuae/falcon-7b
    ```


0. Below are a number of questions from the HellaSwag semantics test. Prompt your AI with something like `Read the following Question carefully and respond with the correct answer from the Options provided:` then copy in an example below. These should be easy for you, but how often does your model get it right?

    ``` 
    Question: A lady walks to a barbell. She bends down and grabs the pole. the lady
    Options: ["pulls a rope attached to the barbell.", "pulls the barbell forward.", "stands and lifts the weight over her head.", "swings and lands in her arms."]
    ```

    ```
    Question: Two women in a child are shown in a canoe while a man pulls the canoe while standing in the water, with other individuals visible in the background. the child and a different man
    Options: ["are driving the canoe, they go down the river flowing side to side.", "walking go down the rapids, while the man in his helicopter almost falls and goes out of canoehood.", "sit in a canoe while the man paddles.", "are then shown paddling down a river in a boat while a woman talks."]
    ```

    ```
    Question: A boy is running down a track. the boy
    Options: ["gets in a mat.", "lifts his body above the height of a pole.", "stands on his hands and springs.", "runs into a car."]
    ```

    ```
    Question: The boy lifts his body above the height of a pole. The boy lands on his back on to a red mat. the boy
    Options: ["gets up from the mat.", "wiggles out of the mat.", "continues to lift his body over the pole.", "turns his body around on the mat."]
    ```

    ```
    Question: The boy lands on his back on to a red mat. The boy gets up from the mat. the boy
    Options: ["does jump jacks on his stick.", "celebrates by clapping and flexing both arms.", "starts doing spins.", "is dancing on the mat."]
    ```

    ```
    Question: A man is standing in front of a camera. He starts playing a harmonica for the camera. he
    Options: ["rocks back and forth to the music as he goes.", "seems to be singing while playing the harmonica.", "painted a fence in front of the camera.", "begins to play the harmonica with his body while looking at the camera."]
    ```

    ```
    Question: A cartoon animation video is shown with people wandering around and rockets being shot. two men
    Options: ["fight robots of evil and ends with a to be continued.", "look in the cameraman's eye and smile.", "push a child in a speedboat in the water.", "are then shown in closeups shooting a shot put."]
    ```

    ```
    Question: A man is holding a pocket knife while sitting on some rocks in the wilderness. then he
    Options: ["opens a can of oil put oil on the knife, and puts oil on a knife and press it through a can filled with oil then cuts several pieces from the sandwiches.", "uses the knife to shave his leg.", "takes a small stone from the flowing river and smashes it on another stone.", "sand the rocks and tops them by using strong pressure."]
    ```

    ```
    Question: Then he takes a small stone from the flowing river and smashes it on another stone. He starts to crush the small stone to smaller pieces. he
    Options: ["starts to party with them and throw the pieces by hand while they celebrate.", "eventually brings it back into view and adds it to the smaller ones to make a small triangular shaped piece.", "cuts the center stone in half and blow it on to make it bigger.", "grind it hard to make the pieces smaller."]
    ```

0. There are many many different testing regiments to ensure any given model can be rated on a number of different factors, from semantics to creativity to factual accuracy to code generation. However the ultimate test will always come from you, the user.
