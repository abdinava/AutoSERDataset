#lets you run this in terminal:
#python -m ser_autodataset video.mp4

#when call in terminal, runs this file

#lets program comm with system.
import sys

#gets the main function:
from .cli import main

#runs main and ends program, exit code.
sys.exit(main())
