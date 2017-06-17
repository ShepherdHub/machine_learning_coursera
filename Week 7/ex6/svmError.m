function error = svmError(predictions, y)
  
  error = mean(double(predictions ~= y));
  
end