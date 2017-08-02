classdef EstimationProgress < handle
  % Progress management for estimation
  %
  % 
  
  % David Kelley, 2017
  
  properties
    theta
    logli 
    thetaIter = 0;
    solverIter = 1;
    
    visible = true;
    
    a
    alpha
  end
  
  properties (Hidden)    
    thetaHist
    likelihoodHist
    solverHist
    
    totalEvaluations = 0;
    nanIterations = 0;    
    
    % Figure components
    figHandle
    axVal
    plotVal
    axPt
    plotPt
    axA
    plotState
    stateSelection 
    
    totalEvalText
    nanItersText
    
    % Others
    theta0
    winSize = [725 700];   
    
    m     % State dimension
  end
  
  properties (SetAccess = protected)
    stopStatus = '';
  end
  
  methods
    %% Constructor
    function obj = EstimationProgress(theta0, visible, m)

      % Set up tracking values
      obj.theta0 = theta0;
      obj.m = m;
      
      maxPossibleIters = 10000;
      obj.thetaHist = nan(maxPossibleIters, length(obj.theta0));
      obj.likelihoodHist = nan(maxPossibleIters, 1);
      obj.solverHist = nan(maxPossibleIters, 1);
      
      obj.thetaIter = 0;
      obj.solverIter = 1;
      
      % Set up figure
      if nargin >= 1 && ~visible
        % Explicitly invisible figure
        obj.visible = false;
      else
        % But default to visible figure
        obj.figHandle = EstimationProgress.getWindow();
        if isempty(obj.figHandle) || ~isvalid(obj.figHandle)
          obj.figHandle = obj.createWindow();
        end
        obj.visible = true;
        
        obj.setupWindow();
        obj.setupPlots();
      end
      
    end
  end
  
  methods
    function stopCond = update(obj, x, optimValues)
      % Update after each iteration
      %   theta: estimate of parameters
      %   optimValues: structure 
      %   state: string of whats going on

      % Update current values
      obj.theta = x;
      obj.logli = -optimValues.fval;
      
      % Update tracking values
      obj.thetaIter = obj.thetaIter + 1;
      obj.thetaHist(obj.thetaIter, :) = x';
      obj.likelihoodHist(obj.thetaIter) = -optimValues.fval;
      obj.solverHist(obj.thetaIter) = obj.solverIter;
      
      % Return indicator of stop condition for optimization functions
      stopCond = ~isempty(obj.stopStatus);
      
      % Update plot
      if obj.visible
        obj.updateFigure();
      end
    end
    
    function updateFigure(obj)
      % Update the figure display with the current object properties
      
      % Current theta value
      obj.plotPt.YData = obj.theta';
      
      % Likelihood history
      ll_XData = find(obj.solverHist == obj.solverIter);
      ll_YData = obj.likelihoodHist(obj.solverHist == obj.solverIter);
      if ~isempty(ll_XData)
        set(obj.plotVal, 'XData', ll_XData, 'YData', ll_YData);
        obj.axVal.XLim(2) = ll_XData(end);
        obj.axVal.Title.String = sprintf('Current Log-likelihood: %9.4f', ll_YData(end));
      end
      
      % Plot of state
      obj.plotStateEstimate(obj.stateSelection.Value);
      
      % Update string of iterations
      obj.totalEvalText.String = obj.getTotalEvalsString();
      obj.nanItersText.String = obj.getNanItersString();
      
      % Draw plot
      drawnow;
    end
    
    function nextSolver(obj)
      obj.solverIter = obj.solverIter + 1;
      
      % Reset stop condition for solver
      if strcmpi(obj.stopStatus, 'stop')
        obj.stopStatus = '';
      end
      
      % Update likelihood value plot
      if obj.visible
        oldline = copyobj(obj.plotVal, obj.axVal);
        % Set markers off, color line based on solver
        oldline.Marker = 'none';
        oldline.LineStyle = '-';
        oldline.Color = oldline.MarkerFaceColor;

        set(obj.plotVal, 'Xdata',[], 'Ydata',[]);
        obj.plotVal.MarkerFaceColor = obj.getCurrentColor();
      end
    end
    
    function dOut = diagnostics(obj)
      % Get diagnostic information about estimation progression
      
      dOut = struct();
      dOut.thetaHist = obj.thetaHist(1:obj.thetaIter, :);
      dOut.likelihoodHist = obj.likelihoodHist(1:obj.thetaIter);
      dOut.solverHist = obj.solverHist(1:obj.thetaIter);
      
      dOut.nanIterations = obj.nanIterations;
    end
    
  end
    
  methods (Hidden)
    function newwin = createWindow(obj)
      % Create a new progress window (when one doesn't exist)   
      screenSize = get(0, 'ScreenSize');
      
      positionVec = [screenSize(3) ./ 2 - 0.5 * obj.winSize(1), ...
        screenSize(4) ./ 2 - 0.5 * obj.winSize(2), ...
        obj.winSize];
      
      newwin = figure('Color', ones(1,3), ...
        'Position', positionVec, ...
        'Name', 'StateSpaceEstimation Progress', ...
        'NumberTitle', 'off', 'Resize', 'off', ...
        'MenuBar', 'none', 'DockControls', 'off', ...
        'HandleVisibility', 'off');
      newwin.Position(3:4) = obj.winSize;

      % Set persistent value
      EstimationProgress.getWindow(newwin);
    end
    
    function setupWindow(obj)
      % Create basic figure elements
      
      newwin = EstimationProgress.getWindow();
      delete(newwin.Children);
      
      % Create a stop button for the solver
      stopBtnXYLoc = [5 5];
      stopBtn = uicontrol('string', 'Stop Current Solver', ...
        'Position', [stopBtnXYLoc 50 20], 'callback', @buttonStop, ...
        'Parent', newwin);
      % Make sure the full text of the button is shown
      stopBtnExtent = get(stopBtn, 'Extent');
      stopBtnPos = [stopBtnXYLoc stopBtnExtent(3:4) + [2 2]];
      % Set the position, using the initial hard coded position, if it is long enough
      set(stopBtn, 'Position', max(stopBtnPos, get(stopBtn, 'Position')));
      
      % Create a stop button for the estimation
      stopBtnXYLoc = [5+stopBtnExtent(3)+5 5];
      stopEstimBtn = uicontrol('string', 'Stop Estimation', ...
        'Position', [stopBtnXYLoc 50 20], 'callback', @buttonStopEstim,  ...
        'Parent', newwin);
      % Make sure the full text of the button is shown
      stopBtnExtent = get(stopEstimBtn, 'Extent');
      stopBtnPos = [stopBtnXYLoc stopBtnExtent(3:4) + [2 2]];
      % Set the position, using the initial hard coded position, if it is long enough
      set(stopEstimBtn, 'Position', max(stopBtnPos, get(stopEstimBtn, 'Position')));
      
      % Create plot of current theta
      axH = axes(newwin);
      obj.axPt = subplot(3, 1, 1, axH);
      obj.plotPt = bar(obj.axPt, 1:length(obj.theta0), obj.theta0, 0.65);
      
      % Plot of likelihood improvement
      obj.axVal = subplot(3, 1, 2, copyobj(axH, newwin));
      delete(obj.axVal.Children);
      obj.plotVal = plot(obj.axVal, 0, 0, 'kd', 'MarkerFaceColor', [1 0 1]);
      
      % Plot of state
      obj.axA = subplot(3, 1, 3, copyobj(axH, newwin));
      delete(obj.axA.Children);
      obj.plotState = plot(obj.axA, 0, 0, 'b');
      
      % Create textbox
      obj.totalEvalText = uicontrol('Style', 'text', ...
        'String', obj.getTotalEvalsString(), ...
        'Position', [newwin.Position(3)-200 15 200 25], ...
        'BackgroundColor', ones(1,3), ...
        'HorizontalAlignment', 'left', ...
        'Parent', newwin);
      obj.nanItersText = uicontrol('Style', 'text', ...
        'String', obj.getNanItersString(), ...
        'Position', [newwin.Position(3)-200 0 200 25], ...
        'BackgroundColor', ones(1,3), ...
        'HorizontalAlignment', 'left', ...
        'Parent', newwin);
        
      function buttonStop(~,~)
        if isempty(obj.stopStatus)
          % Only set if no stop condition has been given
          obj.stopStatus  = 'stop';
        end
      end
      function buttonStopEstim(~,~)
        % Set no matter stop condition in case its 'stop'
        obj.stopStatus  = 'stopestim';
      end
    end
    
    function setupPlots(obj)
      % Fill in plots
      
      % Create plot of current theta
      box(obj.axPt, 'off');
      title(obj.axPt, 'Current Theta Value');
      xlabel(obj.axPt, sprintf('Parameters: %d', length(obj.theta0)));
      set(obj.plotPt, 'edgecolor', 'none')
      set(obj.axPt, 'xlim', [0,1 + length(obj.theta0)])
      % Initial point outline
      hold(obj.axPt, 'on');
      outline = bar(obj.axPt, 1:length(obj.theta0), obj.theta0, 0.8);
      outline.EdgeColor = zeros(1,3);
      outline.FaceColor = 'none';
      
      % Plot of likelihood improvement
      box(obj.axVal, 'off');
      title(obj.axVal, sprintf('Current Log-likelihood'));
      xlabel(obj.axVal, 'Iteration');
      ylabel(obj.axVal, 'Log-likelihood')
      obj.plotVal.MarkerFaceColor =  obj.getCurrentColor();      
      
      % Plot of state
      box(obj.axA, 'off');
      titleText = title(obj.axA, sprintf('Filtered State: '));
      obj.axA.Units = 'pixels';
      titleText.Units = 'pixels';
      dropdownX = obj.axA.Position(1) + titleText.Extent(1) + titleText.Extent(3);
      dropdownY = obj.axA.Position(1) + titleText.Extent(2) - titleText.Extent(4);
      obj.stateSelection = uicontrol('Style', 'popupmenu', ...
        'String', arrayfun(@(x) num2str(x), 1:obj.m, 'Uniform', false), ...
        'Parent', obj.axA.Parent, ...
        'Position', [dropdownX dropdownY 50 20], ...
        'Callback', @(dropdown, ~) obj.plotStateEstimate(dropdown.Value));
      xlabel(obj.axA, 't');
    end    
    
    %% Utility getters
    function strOut = getTotalEvalsString(obj)
      strOut = sprintf('Total likelihood evaulations: %d', obj.totalEvaluations);
    end
    
    function strOut = getNanItersString(obj)
      strOut = sprintf('Bad likelihood evaulations: %d', obj.nanIterations);
    end
    
    function currentColor = getCurrentColor(obj)
      currentColor = [mod(obj.solverIter - 1, 2), 0, 1];
    end
    
    %% Plot functions
    function plotStateEstimate(obj, stateIndex)
      % Plot the filtered/smoothed state
      
      if isempty(obj.alpha)
        stateEstimate = obj.a;
        title(obj.axA, 'Filtered State:');
      else
        stateEstimate = obj.alpha;
        oldTitlePos = obj.axA.Title.Extent;
        titleObj = title(obj.axA, 'Smoothed State:');
        titleObj.Position(1) = titleObj.Position(1) - ...
          (titleObj.Extent(3) - oldTitlePos(3));        
      end
      
      if ~isempty(stateEstimate)
        plotY = stateEstimate(stateIndex,:)';
      else
        plotY = [];
      end
      
      set(obj.plotState, 'Xdata', 1:length(plotY), 'Ydata', plotY);
      if ~isempty(stateEstimate)
        obj.axA.XLim = [1 length(plotY)];
      end
      
      % Set axes
      if obj.axA.YLim(2) < max(plotY)
        obj.axA.YLim(2) = max(plotY);
      end
      if obj.axA.YLim(1) > min(plotY)
        obj.axA.YLim(1) = min(plotY);
      end
    end    
  end
  
  methods (Static)
    % Static persistent window
    function window = getWindow(newwindow)
      % Get the handle to the figure window
      persistent window_persistent
      if nargin >= 1 && ~isempty(newwindow)
        window_persistent = newwindow;
      end
      
      window = window_persistent;
    end
  end
  
end