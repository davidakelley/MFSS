classdef StateSpaceEstimation < AbstractStateSpace
  % Maximum likelihood estimation of parameters 
  % -------------------------------------------
  %   ss_estimated = ss.estimate(y, ss0)
  %
  % Estimate parameter values of a state space system. Indicate values to
  % be estimated by setting them to nan. An initial set of parameters must
  % be provided in ss0.
  %
  % Pseudo maximum likelihood estimation
  % ------------------------------------
  %   ss_estimated = ss.em_estimate(y, ss0)
  %
  % Initial work on implementing a general EM algorithm.
  
  % David Kelley, 2016-2017
  %
  % TODO (1/17/17)
  % ---------------
  %   - EM algorithm
  
  properties
    % Screen output during ML estimation
    verbose = false;
    diagnosticPlot = true;
    
    solver = {'fmincon', 'fminsearch'};
    solveIterMax = 20;
    
    % ML-estimation tolerances
    tol = 1e-10;      % Final estimation tolerance
    stepTol = 1e-8;  % Step-size tolerance for ML theta elements
    solveTol = 1e-11;   % EM tolerance
    
    % ML Estimation parameters
    ThetaMapping      % Mapping from theta vector to parameters
    useGrad = false;   % Indicator for use of analytic gradient
    
    fminsearchMaxIter = 500;
  end
  
  properties (Hidden=true)
    progressWin
  end
  
  methods
    %% Constructor
    function obj = StateSpaceEstimation(Z, d, H, T, c, R, Q, varargin)
      obj = obj@AbstractStateSpace(Z, d, H, T, c, R, Q);
      
      inP = inputParser;
      inP.addParameter('a0', [], @isnumeric);
      inP.addParameter('P0', [], @isnumeric);
      inP.addParameter('LowerBound', [], @(x) isa(x, 'StateSpace'));
      inP.addParameter('UpperBound', [], @(x) isa(x, 'StateSpace'));
      inP.addParameter('ThetaMap', [], @(x) isa(x, 'ThetaMap'));
      inP.parse(varargin{:});
      inOpts = inP.Results;

      % Initial ThetaMap generation - will be augmented to include restrictions
      if ~isempty(inOpts.ThetaMap)
        obj.ThetaMapping = inOpts.ThetaMap;
      else
        obj.ThetaMapping = ThetaMap.ThetaMapEstimation(obj);
      end
      
      % Initial values - If initial values are not passed they will be set as 
      % the default values at each iteration of the estimation, effectively 
      % making them a function of other parameters. 
      if ~isempty(inOpts.a0) && ~isempty(inOpts.P0)
        obj = obj.setInitial(inOpts.a0, inOpts.P0);
      elseif ~isempty(inOpts.a0) 
        obj = obj.setInitial(inOpts.a0);
      end
      
      obj.ThetaMapping = obj.ThetaMapping.addRestrictions(inOpts.LowerBound, inOpts.UpperBound);
      
      % Estimation restrictions - estimation will be bounded by bounds on
      % parameter matricies passed and non-negative restrictions on variances. 
%       obj = obj.generateRestrictions(inOpts.LowerBound, inOpts.UpperBound);
    end
    
    %% Estimation methods
    function [ss_out, diagnostic, gradient] = estimate(obj, y, ss0)
      % Estimate missing parameter values via maximum likelihood.
      %
      % ss = ss.estimate(y, ss0) estimates any missing parameters in ss via
      % maximum likelihood on the data y using ss0 as an initialization.
      %
      % ss.estimate(y, ss0, a0, P0) uses the initial values a0 and P0
      %
      % [ss, flag] = ss.estimate(...) also returns the fmincon flag
                 
      assert(isnumeric(y), 'y must be numeric.');
      assert(isa(ss0, 'StateSpace'));
      
      assert(obj.ThetaMapping.nTheta > 0, 'All parameters known. Unable to estimate.');
      
      % Initialize
      obj.checkConformingSystem(ss0);

      theta0 = obj.ThetaMapping.system2theta(ss0);
      assert(all(isfinite(theta0)), 'Non-finite values in starting point.');
      
      if obj.diagnosticPlot
        % Close all old estimation plots
        estimWins = findobj('Tag', 'StateSpaceEstimationWindow');
        close(estimWins(2:end));
        obj.progressWin = StateSpaceEstimation.initFigure(theta0, obj.m, estimWins);
      end
      
      % Run fminunc/fmincon
      minfunc = @(theta) obj.minimizeFun(theta, y);
      nonlconFn = @obj.nlConstraintFun;
      
      if obj.verbose
        displayType = 'iter-detailed';
      else
        displayType = 'none';
      end
      
      optFMinCon = optimoptions(@fmincon, ...
        'Algorithm', 'interior-point', ...
        'SpecifyObjectiveGradient', obj.useGrad, ...
        'UseParallel', false, ...
        'Display', displayType, ...
        'MaxFunctionEvaluations', 50000, ...
        'MaxIterations', 1000, ...
        'FunctionTolerance', obj.tol, ...
        'OptimalityTolerance', obj.tol, ...
        'StepTolerance', obj.stepTol, ...
        'OutputFcn', @outfun, ...
        'TolCon', 0);
      
      optFMinUnc = optimoptions(@fminunc, ...
        'Algorithm', 'quasi-newton', ...
        'SpecifyObjectiveGradient', obj.useGrad, ...
        'UseParallel', false, ...
        'Display', displayType, ...
        'MaxFunctionEvaluations', 50000, ...
        'MaxIterations', 1000, ...
        'FunctionTolerance', obj.tol, ...
        'OptimalityTolerance', obj.tol, ...
        'StepTolerance', obj.stepTol, ...
        'OutputFcn', @outfun);
      
      if all(strcmpi(obj.solver, 'fminsearch')) && ...
          (ischar(obj.fminsearchMaxIter) &&  strcmpi(obj.fminsearchMaxIter, 'default'))
        searchMaxIter = 200 * obj.ThetaMapping.nTheta;
      else
        searchMaxIter = obj.fminsearchMaxIter;
      end
      
      optFMinSearch = optimset('Display', displayType, ...
        'MaxFunEvals', 5000 * obj.ThetaMapping.nTheta, ...
        'MaxIter', searchMaxIter, ...
        'OutputFcn', @outfun);
      
      warning off MATLAB:nearlySingularMatrix;
      
      solverFun = obj.solver;
      
      % Save history of theta values
      nTheta = size(theta0, 1);
      if iscell(obj.solver)
        maxPossibleIters = max(optFMinUnc.MaxIterations, optFMinSearch.MaxIter) * obj.solveIterMax;
      else
        switch obj.solver
          case {'fminunc', 'fmincon'}
            maxPossibleIters = optFMinUnc.MaxIterations * obj.solveIterMax;
          case 'fminsearch'
            maxPossibleIters = optFMinSearch.MaxIter * obj.solveIterMax;
        end
      end
      
      thetaIter = 0;
      thetaHist = nan(maxPossibleIters, nTheta);
      likelihoodHist = nan(maxPossibleIters, 1);
      solverIter = nan(maxPossibleIters, 1);
      
      loopContinue = true;
      iter = 0; logli = []; stopestim = false;
      while loopContinue
        iter = iter + 1;
        logli0 = logli;
        if iscell(obj.solver)
          solverFun = obj.solver{mod(iter+1, length(obj.solver))+1};
        end
        
        switch solverFun
          case 'fminunc'
            [thetaHat, logli, ~, ~, gradient] = fminunc(...
              minfunc, theta0, optFMinUnc);
          case 'fmincon'
            [thetaHat, logli, ~, ~, ~, gradient] = fmincon(... 
              minfunc, theta0, [], [], [], [], [], [], nonlconFn, optFMinCon);
          case 'fminsearch'
            tempGrad = obj.useGrad;
            obj.useGrad = false;
            minfunc = @(theta) obj.minimizeFun(theta, y);

            [thetaHat, logli, ~] = fminsearch(...
              minfunc, theta0, optFMinSearch);
            obj.useGrad = tempGrad;
            minfunc = @(theta) obj.minimizeFun(theta, y);
            
            if obj.filterUni
              ssHat = obj.ThetaMapping.theta2system(thetaHat);
              ssHat.filterUni = false;
              [~, trueLogli] = ssHat.filter(y);
              logli = -trueLogli;
            end
            
            gradient = [];
        end
        
        loopContinue = ~stopestim && iter < obj.solveIterMax && ...
          (iter <= 2 || abs(logli0 - logli) > obj.solveTol);
        
        if logli0 < logli
          warning('StateSpaceEstimation:estimate:solverDecrease', ...
            ['Solver decreased likelihood by %g. \n' ...
            'Returning higher likelihood solution.'], logli - logli0);
          thetaHat = thetaHist(find(likelihoodHist == logli0, 1, 'last'), :)';
          assert(~isempty(thetaHat));
          logli = logli0;
          gradient = [];
          loopContinue = false;
        end
        
        assert(~any(isnan(thetaHat)), 'Estimation Error');
        
        theta0 = thetaHat;
      end
      
      warning on MATLAB:nearlySingularMatrix;
  
      % Save estimated system to current object
      ss_out = obj.ThetaMapping.theta2system(thetaHat);
      
      if obj.diagnosticPlot
        % Run smoother, plot smoothed state
        alpha = ss_out.smooth(y);
        setappdata(obj.progressWin, 'a', alpha);
        
        % Retitle plot and move title so it doesn't overlap dropdown
        filterPlot = findobj('Tag', 'sseplotfiltered');
        oldTitlePos = filterPlot.Parent.Title.Extent;
        titleObj = title(filterPlot.Parent, 'Smoothed State:');
        titleObj.Position(1) = titleObj.Position(1) - ...
          (titleObj.Extent(3) - oldTitlePos(3));
        
        dropdown = findobj('Tag', 'sseplotfiltered_dropdown');
        StateSpaceEstimation.plotFilteredCallback(dropdown, []);
      end
      
      % Create diagnostic output
      thetaHist(thetaIter+1:end,:) = [];
      likelihoodHist(thetaIter+1:end) = [];
      solverIter(thetaIter+1:end) = [];
      diagnostic = AbstractSystem.compileStruct(thetaHist, likelihoodHist, solverIter);
      
      % Nested function for getting diagnostic output
      function stop = outfun(x,optimValues,state)
        thetaIter = thetaIter + 1;
        thetaHist(thetaIter, :) = x';
        likelihoodHist(thetaIter) = optimValues.fval;
        solverIter(thetaIter) = iter;
        stop = false;
        
        if obj.diagnosticPlot
          if strcmpi(getappdata(gcf, 'stop_condition'), 'stop')
            stop = true;
            setappdata(gcf, 'stop_condition', '');
            state = 'done';
          end
          if strcmpi(getappdata(gcf, 'stop_condition'), 'stopestim')
            stop = true;
            stopestim = true;
            state = 'done';
          end
          
          optimValues.iteration = thetaIter;
          
          StateSpaceEstimation.sseplotpoint(x, optimValues, state);
          StateSpaceEstimation.sseplotvalue(x, optimValues, state, iter);
          StateSpaceEstimation.sseplotfiltered([], [], state);
          drawnow;
        end
      end
    end
    
    function ss0 = em_estimate(obj, y, ss0, varargin)
      % Estimate parameters through pseudo-maximum likelihood EM algoritm
      
      % See Greene's Econometric Analysis for the general setup here. 
      % The approach is limited in that it cannot enforce bound conditions or
      % nonlinear restrictions on parameter elements. 
      %
      % We generate a matrix F and a vector G such that F * beta = G where beta
      % will be either the parameter matrix Z or T. 
      %
      % For states with a fixed (or zeros) error variance... what do we do?
      
      % TODO: Remove time invariance restriction.
      assert(obj.timeInvariant, ...
        'EM Algorithm only developed for time-invariant cases.');
      
      [obj, ss0] = obj.checkConformingSystem(y, ss0, varargin{:});
      
      % Generate F and G matricies for state and observation equations
      % Does it work to run the restricted OLS for the betas we know while
      % letting the variances be free in the regression but only keeping
      % the ones we're estimating afterward? I think so.
      Fobs = [];
      Gobs = [];
      Fstate = [];
      Gstate = [];
      
      % Iterate over EM algorithm
      iter = 0; logli0 = nan; gap = nan;
      while iter < 2 || gap > obj.tol
        iter = iter + 1;
        
        % E step: Estiamte complete log-likelihood
        [alpha, sOut, fOut] = ss0.smooth(y);
        [V, J] = ss0.getVariances(y, alpha, sOut, fOut);
        
        % M step: Get parameters that maximize the likelihood
        % Measurement equation
        [ss0.Z, ss0.d, ss0.H] = obj.restrictedOLS(y', alpha', V, J, Fobs, Gobs);
        % State equation
        [ss0.T, ss0.c, RQR] = obj.restrictedOLS(...
          alpha(:, 2:end)', alpha(:, 1:end-1)', V, J, Fstate, Gstate);
        
        % Report
        if obj.verbose
          gap = abs((sOut.logli - logli0)/mean([sOut.logli; logli0]));
        end
        logli0 = sOut.logli;
      end
    end
    
    function obj = setInitial(obj, a0, P0)
      % Set a0 and P0 for the system
      
      % Set the avaliable values
      obj = setInitial@AbstractStateSpace(obj, a0, P0);
      
      % Make sure the ThetaMap gets updated
      obj.ThetaMapping = obj.ThetaMapping.updateInitial(a0, P0);      
    end
  end
  
  methods (Hidden = true)
    %% Maximum likelihood estimation helper methods
    function [negLogli, gradient] = minimizeFun(obj, theta, y)
      % Get the likelihood of
      ss1 = obj.ThetaMapping.theta2system(theta);
      ss1.filterUni = obj.filterUni;
      
      if any(diag(ss1.H) < 0)
        keybaord;
      end
      
      % Really enforce constraints
      if any(obj.nlConstraintFun(theta) > 0)
        % warning('Constraint violated in solver.');
        negLogli = 1e30 * max(obj.nlConstraintFun(theta));
        gradient = nan(size(theta));
        return
      end
      
      try
        if obj.useGrad
          [rawLogli, rawGradient] = ss1.gradient(y, obj.ThetaMapping);
          a = [];
        else
          [a, rawLogli] = ss1.filter(y);
          rawGradient = [];
        end
        
        saveLogli = getappdata(obj.progressWin, 'logli');
        if isempty(saveLogli) || rawLogli > saveLogli
          setappdata(obj.progressWin, 'logli', rawLogli);
          setappdata(obj.progressWin, 'a', a);
        end
      catch
        rawLogli = nan;
        rawGradient = nan(obj.ThetaMapping.nTheta, 1);
      end
      
      negLogli = -rawLogli;
      gradient = -rawGradient;
    end
    
    function [cx, ceqx, deltaCX, deltaCeqX] = nlConstraintFun(obj, theta)
      % Constraints of the form c(x) <= 0 and ceq(x) = 0.
      scale = 1e6;
      vec = @(M) reshape(M, [], 1);
      
      % Return the negative determinants in cx
      ss1 = obj.ThetaMapping.theta2system(theta);
      % cx = scale * -[det(ss1.H) det(ss1.Q)]';
      cx = []; % Removed H, Q dets on 1/25
      if ~obj.usingDefaultP0
        cx = [cx; scale * -det(ss1.Q0)];
      end         
      ceqx = 0;
      
      % And the gradients 
      warning off MATLAB:nearlySingularMatrix
      warning off MATLAB:singularMatrix
      G = obj.ThetaMapping.parameterGradients(theta);
    	deltaCX = scale * -[det(ss1.H) * G.H * vec(inv(ss1.H)), ...
                 det(ss1.Q) * G.Q * vec(inv(ss1.Q))];
      if ~obj.usingDefaultP0
        [~, G.P0] = obj.ThetaMapping.initialValuesGradients(theta);
        % This is suspect. Need to rewrite G.P0 first.
        deltaCX = [deltaCX det(ss1.Q0) * G.P0 * vec(ss1.R0 / ss1.Q0 * ss1.R0')];
      end
      warning on MATLAB:nearlySingularMatrix
      warning on MATLAB:singularMatrix
      
      deltaCeqX = sparse(0);
      
      % Constrain a submatrix of T to be stationary: 
      % c(x) should be abs(max(eig(T))) - 1
      if ~isempty(obj.stationaryStates)
        eigs = eig(ss1.T(obj.stationaryStates,obj.stationaryStates,obj.tau.T(1)));
        stationaryCx = scale * (max(abs(eigs)) - 1);
        cx = [cx; stationaryCx];
        deltaCX = [];
      end
      
      if any(isnan(cx))
        warning('Nan constraint.');
      end
    end
  end
  
  methods (Static = true, Hidden = true)
    %% Plot functions for ML estimation
    function diagnosticF = initFigure(theta0, m, oldFig)
      % Create diagnostic figure
      screenSize = get(0, 'ScreenSize');
      winSize = [725 700];
      if nargin > 2 && ~isempty(oldFig)
        diagnosticF = oldFig;
        delete(diagnosticF.Children);
        delete(findobj('Tag', 'sseplotfiltered_dropdown'));
        setappdata(diagnosticF, 'a', []);
        setappdata(diagnosticF, 'logli', []);
        setappdata(diagnosticF, 'stop_condition', '');
      else
        diagnosticF = figure('Color', ones(1,3), ...
          'Tag', 'StateSpaceEstimationWindow', ...
          'Position', ...
          [screenSize(3)./2-0.5*winSize(1), screenSize(4)./2-0.5*winSize(2), winSize], ...
          'Name', 'StateSpaceEstimation Progress', ...
          'NumberTitle', 'off');
        diagnosticF.Position(3:4) = [725 700];
      end
      
      axH = axes(diagnosticF);
      axPt = subplot(3, 1, 1, axH);
      axVal = subplot(3, 1, 2, copyobj(axH, diagnosticF));
      axA = subplot(3, 1, 3, copyobj(axH, diagnosticF));
      
      StateSpaceEstimation.sseplotpoint(theta0, [], 'setup', 0, axPt);
      StateSpaceEstimation.sseplotvalue(theta0, [], 'setup', 0, axVal);
      StateSpaceEstimation.sseplotfiltered([], [], 'setup', m, axA);
      
      % Create a stop button for the solver
      stopBtnXYLoc = [2 5];
      stopBtn = uicontrol('string', 'Stop Current Solver', ...
        'Position',[stopBtnXYLoc 50 20],'callback',@buttonStop);
      % Make sure the full text of the button is shown
      stopBtnExtent = get(stopBtn,'Extent');
      stopBtnPos = [stopBtnXYLoc stopBtnExtent(3:4)+[2 2]]; % Read text extent of stop button
      % Set the position, using the initial hard coded position, if it is long enough
      set(stopBtn,'Position',max(stopBtnPos,get(stopBtn,'Position')));
      
      function buttonStop(~,~)
        setappdata(gcf,'stop_condition','stop');
      end
      
      % Create a stop button for the estimation
      stopBtnXYLoc = [2+stopBtnExtent(3)+2 5];
      stopEstimBtn = uicontrol('string', 'Stop Estimation', ...
        'Position',[stopBtnXYLoc 50 20],'callback', @buttonStopEstim);
      % Make sure the full text of the button is shown
      stopBtnExtent = get(stopEstimBtn,'Extent');
      stopBtnPos = [stopBtnXYLoc stopBtnExtent(3:4)+[2 2]]; % Read text extent of stop button
      % Set the position, using the initial hard coded position, if it is long enough
      set(stopEstimBtn,'Position',max(stopBtnPos,get(stopEstimBtn,'Position')));
      
      function buttonStopEstim(~,~)
        setappdata(gcf,'stop_condition','stopestim');
      end
    end
    
    function stop = sseplotpoint(x, ~, state, varargin)
      stop = false;
      
      persistent plotx
      if strcmpi(state, 'setup')
        axPt = varargin{2};
        
        plotx = bar(axPt, 1:length(x), x, 0.65);
        box(axPt, 'off');
        title(axPt, 'Current Point');
        xlabel(axPt, sprintf('Varaibles: %d', length(x)));
        set(plotx,'edgecolor','none')
        set(plotx.Parent,'xlim',[0,1 + length(x)])
        
        set(plotx,'Tag','sseplotx');
        
        % Make initial point outline
        hold(axPt, 'on');
        outline = bar(axPt, 1:length(x), x, 0.8);
        outline.EdgeColor = zeros(1,3);
        outline.FaceColor = 'none';        
        return
      end
      
      if ~strcmpi(state, 'iter')
        return
      end
      
      set(plotx,'Ydata',x);
    end
    
    function stop = sseplotvalue(~, optimValues, state, varargin)
      % We're plotting the likelihood, not the function value
      % (AKA, reverse the sign on everything).
      
      stop = false;
      if ~strcmpi(state, {'setup', 'init', 'iter', 'done'})
        error('Unsuported plot command.');
      end
      
      iter = varargin{1};
      currentColor = [mod(iter, 2), 0, 1];
      
      persistent plotfval
      if strcmpi(state, 'setup')
        axH = varargin{2};
        
        plotfval = plot(axH, 0, 0, 'kd', 'MarkerFaceColor', [1 0 1]);
        box(axH, 'off');
        title(axH, sprintf('Current Log-likelihood:'));
        xlabel(axH, 'Iteration');
        ylabel(axH, 'Log-likelihood')
        
        set(plotfval, 'Tag', 'sseplotfval');
        
        plotfval.MarkerFaceColor = currentColor;
        return
      end
      
      iteration = optimValues.iteration;
      llval = -optimValues.fval;
      
      if strcmpi(state, 'init')
        set(plotfval,'Xdata', iteration, 'Ydata', llval);
        
        plotfval.MarkerFaceColor = currentColor;
        return
      end
      
      newX = [get(plotfval,'Xdata') iteration];
      newY = [get(plotfval,'Ydata') llval];
      set(plotfval,'Xdata',newX, 'Ydata',newY);
      set(get(plotfval.Parent,'Title'),'String', ...
        sprintf('Current Function Value: %g', llval));
      
      if plotfval.Parent.YLim(2) < llval
        plotfval.Parent.YLim(2) = llval;
      end
      
      if strcmpi(state, 'done')
        oldline = copyobj(plotfval, plotfval.Parent);
        % Set markers off, color line based on solver
        oldline.Marker = 'none';
        oldline.LineStyle = '-';
        oldline.Color = currentColor;
        
        set(plotfval,'Xdata',[], 'Ydata',[]);
      end      
    end
    
    function stop = sseplotfiltered(~, ~, state, varargin)
      % Plot the filtered state, with a dropdown box for which state
      
      stop = false;
      if ~strcmpi(state, {'setup', 'init', 'iter', 'done'})
        error('Unsuported plot command.');
      end

      if strcmpi(state, 'setup')
        m = varargin{1};
        axH = varargin{2};

        plotfiltered = plot(axH, 0, 0, 'b');
        box(axH, 'off');
        titleText = title(axH, sprintf('Filtered State: '));
        
        plotfiltered.Parent.Units = 'pixels';
        titleText.Units = 'pixels';
        
        dropdownX = plotfiltered.Parent.Position(1) + titleText.Extent(1) + titleText.Extent(3);
        dropdownY = plotfiltered.Parent.Position(1) + titleText.Extent(2) - titleText.Extent(4);       
        uicontrol('Style', 'popupmenu', ...
          'String', arrayfun(@(x) num2str(x), 1:m, 'Uniform', false), ...
          'Position', [dropdownX dropdownY 50 20], ...
          'Callback', @StateSpaceEstimation.plotFilteredCallback, ...
          'Tag', 'sseplotfiltered_dropdown');
        
        xlabel(axH, 't');
        
        set(plotfiltered, 'Tag', 'sseplotfiltered');
        return
      end
      
      if strcmpi(state, 'init')
        return
      end
            
      % Iter and done states
      dropdown = findobj('Tag', 'sseplotfiltered_dropdown');
      StateSpaceEstimation.plotFilteredCallback(dropdown, []);
    end
    
    function plotFilteredCallback(dropdown, ~)
      % Get the filtered state stored in the figure's appdata, plot acording to
      % current value of the dropdown. 
      
      plotState = dropdown.Value;
      a = getappdata(dropdown.Parent, 'a');
      if isempty(a)
        % a not yet set, nothing to plot
        return
      end
      plotY = a(plotState,:);
            
      plotfiltered = findobj('Tag', 'sseplotfiltered');
      set(plotfiltered, 'Xdata', 1:length(plotY), 'Ydata', plotY);
      
      % Set axes
      if plotfiltered.Parent.YLim(2) < max(plotY)
        plotfiltered.Parent.YLim(2) = max(plotY);
      end
      if plotfiltered.Parent.YLim(1) > min(plotY)
        plotfiltered.Parent.YLim(1) = min(plotY);
      end
    end
  end
  
  methods (Hidden = true)
    %% EM Algorithm Helper functions
    function [beta, sigma] = restrictedOLS(y, X, V, J, F, G)
      % Restricted OLS regression
      % See Econometric Analysis (Greene) for details.
      T_dim = size(y, 1);
      
      % Simple OLS
      xxT = sum(V, 3) + X * X';
      yxT = sum(J, 3) + y * X';
      yyT = sum(V, 3) + y * y';
      
      betaOLS = yxT/xxT;
      sigmaOLS = (yyT-OLS*yxT');
      
      % Restricted OLS estimator
      beta = betaOLS - (betaOLS * F - G) / (F' / xxT * F) * (F' / xxT);
      sigma = (sigmaOLS + (betaOLS * F - G) / ...
        (F' / xxT * F) * (betaOLS * F - G)') / T_dim;
    end
  end
end
