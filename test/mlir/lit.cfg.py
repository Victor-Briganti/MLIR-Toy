import os
import lit.formats

config.name = 'MLIR'
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.mlir', '.toy']

# Define where to look for the tests
config.test_source_root = os.path.dirname(__file__)

# Tool Substitutions
project_root = os.path.dirname(os.path.dirname(config.test_source_root))
toy_compiler = os.path.join(project_root, 'build', 'bin', 'main')

# Substitutions for the RUN lines
config.substitutions.append(('toy-compiler', toy_compiler))
