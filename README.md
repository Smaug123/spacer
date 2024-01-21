# Training a transformer to put spaces into text

Given text like "mynameisozymandiaskingofkings", the model should output `0100010100000000100010100001`, with a `1` for each character that has a space after it.

I tried using ChatGPT 4 a lot to help me make this.
It was a *dismal* failure, but you can still see the remnants of its garbage in the random comments through the code.
Similarly I have had to disable GitHub Copilot.

## What is wrong

As of the commit where I write this (42ca140), I am training the model on a very tiny test set.
Loss correctly decreases during training.
However, during inference (where I use a verbatim example from the training set), I consistently get outputs which are all the same probability at every position, either very high or very low.
Moreover, this probability changes wildly based on the initial value of `tgt`.

I was under the impression that the transformer target mask would prevent such dependencies, but clearly I don't understand something.
It appears that the model has learned to get the correct answer when it is given the correct answer, and the masking has not caused it to be unable to see the correct answer.
