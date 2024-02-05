## Input arguments for running `main.py` script:

### `<script_name> <input_image> <option> <message> <N> <M>`

`<input_image>` - path to image file you want to compress
<br/>
`<option>` - h for hiding message or e for extracting message
<br/>
`<message>` - path to .txt file containing message you want to hide
<br/>
`<N>` - compression threshold
<br/>
`<M>` - number of unique sets of coefficients used in F5 steganography algorithm

<br/>
Example usages:

`python main.py puppy.bmp h message.txt 20 3`
<br/>
`python main.py decomp_puppy.bmp e message.txt 20 3`


See results in `porocilo.pdf`
