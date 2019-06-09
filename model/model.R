# Run at start-up
args <- commandArgs()
if (any(grepl('train', args))) {
    train()}
if (any(grepl('serve', args))) {
    serve()}
