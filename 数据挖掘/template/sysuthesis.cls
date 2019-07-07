%%
%% This is file `sysuthesis.cls',
%% customizing `book.cls' from \LaTeXe, which does
%% \ProvidesClass{book}
%%               [2007/10/19 v1.4h
%%  Standard LaTeX document class]
%% and was
%% generated with the docstrip utility.
%%
%% The original source files were:
%% classes.dtx  (with options: `book')
%% Some modifications are from references to files
%% `sysubook.cls', `ucthesis.cls', and `uiucthesis.cls'
%% in their correspending packages.
%% I decide not to rely on KOMA-script bundle, but to copy
%% some of its code.
%%
%% Copyright (C) 2009 Alexander Huang <loligo@sohu.com>.
%% --------------------------------------------------------------
%% 
%% It may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License, either version 1.3c
%% of this license or (at your option) any later version.
%% The latest version of this license is in
%%    http://www.latex-project.org/lppl.txt
%% and version 1.3c or later is part of all distributions of LaTeX
%% version 2005/12/01 or later.
%% 
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{sysuthesis}
              [2010/02/28 v1.0
 Sun Yat-sen University thesis class]
\DeclareOption{PhD}{\def\@degree{\CJK@Doctor} \def\@thesis{\CJK@Degree}}
\DeclareOption{MSc}{\def\@degree{\CJK@Master} \def\@thesis{\CJK@Degree}}
\DeclareOption{BSc}{\def\@degree{\CJK@Bachelor} \def\@thesis{\CJK@Graduate}}
\DeclareOption*{\PackageWarning{sysuthesis}{Unknown option `\CurrentOption'}}
\ExecuteOptions{MSc}
%\ExecuteOptions{BSc}
\ProcessOptions\relax
\LoadClass[titlepage,a4paper,12pt]{book}[2007/10/19]
%\input{bk1\@ptsize.clo}
%\setlength\lineskip{1\p@}
%\setlength\normallineskip{1\p@}
\renewcommand\baselinestretch{1.5}
%\setlength\parskip{0\p@ \@plus \p@}
%\@lowpenalty   51
%\@medpenalty  151
%\@highpenalty 301
%\setcounter{topnumber}{2}
%%%% allow more loose floating to avoid a page with only a float.
\renewcommand\topfraction{.90}
%\setcounter{bottomnumber}{1}
\renewcommand\bottomfraction{.65}
%\setcounter{totalnumber}{3}
%\renewcommand\textfloatsep{20pt plus2pt minus4pt}
\renewcommand\textfraction{.10}
\renewcommand\floatpagefraction{.20}
%\setcounter{dbltopnumber}{2}
%\renewcommand\dbltopfraction{.7}
%\renewcommand\dblfloatpagefraction{.5}
%\newcommand{\headrule}{\\[1pt]\rule{\textwidth}{0.5pt}\vskip-0.5pt}
\newcommand{\headrule}{\hrule height0.4pt \vskip-0.5pt}
\if@twoside
	\def\ps@headings{%
		\let\@oddfoot\@empty\let\@evenfoot\@empty
		% If caption too long, uncomment the next line:
		%\iftrue{\def\leftmark{\ } \def\rightmark{\ }}\fi
		\def\@evenhead{\vbox{\thepage\hspace{2em}{%
			\sffamily\small\thesisname}%
			\hfill{\small\leftmark}\headrule}}%
		\def\@oddhead{\vbox{{\def\\{}\sffamily\small\@title}%
			\hspace{2em}{%
			\small\rightmark}\hfill\thepage\headrule}}%
		\let\@mkboth\markboth
		\def\chaptermark##1{%
			\markboth {%
			\ifnum \c@secnumdepth >\m@ne \if@mainmatter
			\chapternamethechapter{\thechapter}\ %
			\fi\fi
		##1}{}}%
		\def\sectionmark##1{%
			\markright {%
			\ifnum \c@secnumdepth >\z@
			\thesection\ \ %
			\fi
		##1}}}
\else
	\def\ps@headings{%
		\let\@oddfoot\@empty
		\def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
		\let\@mkboth\markboth
		\def\chaptermark##1{%
			\markright {%
			\ifnum \c@secnumdepth >\m@ne \if@mainmatter
			\chapternamethechapter{\thechapter}\ %
			\fi\fi
	##1}}}
\fi

\renewcommand \thepart {\@arabic\c@part}

\def\@part[#1]#2{%
    \ifnum \c@secnumdepth >-2\relax
      \refstepcounter{part}%
      \addcontentsline{toc}{part}{\thepart\hspace{1em}#1}%
    \else
      \addcontentsline{toc}{part}{#1}%
    \fi
    \markboth{}{}%
    {\centering
     \interlinepenalty \@M
     \normalfont
     \ifnum \c@secnumdepth >-2\relax
       \huge\bfseries \CJKorder~\thepart~\partname
       \par
       \vskip 20\p@
     \fi
     \Huge \bfseries #2\par}%
    \@endpart}
\def\@spart#1{%
    {\centering
     \interlinepenalty \@M
     \normalfont
     \Huge \bfseries #1\par}%
    \@endpart}
%\def\@endpart{\vfil\newpage
%              \if@twoside
%               \if@openright
%                \null
%                \thispagestyle{empty}%
%                \newpage
%               \fi
%              \fi
%              \if@tempswa
%                \twocolumn
%              \fi}
%\newcommand\chapter{\if@openright\cleardoublepage\else\clearpage\fi
%                    \thispagestyle{plain}%
%                    \global\@topnum\z@
%                    \@afterindentfalse
%                    \secdef\@chapter\@schapter}
\def\@chapter[#1]#2{\ifnum \c@secnumdepth >\m@ne
                       \if@mainmatter
                         \refstepcounter{chapter}%
                         \typeout{\@chapapp\space\thechapter.}%
                         \addcontentsline{toc}{chapter}%
                                   {\protect\numberline{%
			\chapternamethechapter{\thechapter}}#1}%
                       \else
                         \addcontentsline{toc}{chapter}{\heiti #1}%
                       \fi
                    \else
                      \addcontentsline{toc}{chapter}{\heiti #1}%
                    \fi
                    \chaptermark{#1}%
                    \addtocontents{lof}{\protect\addvspace{10\p@}}%
                    \addtocontents{lot}{\protect\addvspace{10\p@}}%
                    \if@twocolumn
                      \@topnewpage[\@makechapterhead{#2}]%
                    \else
                      \@makechapterhead{#2}%
                      \@afterheading
                    \fi}
\def\@makechapterhead#1{%
	\vspace*{\skip@beforechapter}%
	{\centering\normalfont\size@chapter
	\ifnum \c@secnumdepth >\m@ne\if@mainmatter
		\Huge\sffamily\bfseries \chapternamethechapter{\thechapter}\ \ 
	\fi\fi
	\interlinepenalty\@M
	\Huge \bfseries #1\par\nobreak
	\vskip \skip@afterchapter}}

\def\@makeschapterhead#1{%
	\vspace*{\skip@beforechapter}%
	{\centering\normalfont\size@chapter
	\interlinepenalty\@M
	\Huge \bfseries #1\par\nobreak
	\vskip \skip@afterchapter}}
\renewcommand\section{\@startsection{section}{1}{\parindent@section}%
	{\skip@beforesection}%
	{\skip@aftersection}%
	{\flushleft\normalfont\size@section\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{\parindent@section}%
	{\skip@beforesection}%
	{\skip@aftersection}%
	{\flushleft\normalfont\large\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\parindent@section}%
	{\skip@beforesection}%
	{\skip@aftersection}%
	{\flushleft\normalfont\large}}
\renewcommand\paragraph{\@startsection{paragraph}{4}{\parindent}%
	{\skip@beforeparagraph}%
	{\skip@afterparagraph}%
	{\flushleft\normalfont\large}}
\renewcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
	{\skip@beforeparagraph}%
	{\skip@afterparagraph}%
	{\flushleft\normalfont\large}}

\renewcommand\appendix{\par
  \setcounter{chapter}{0}%
  \setcounter{section}{0}%
  \gdef\@chapapp{\appendixname}%
  \renewcommand\chapternamethechapter[1]{\@chapapp~##1}%
  \gdef\thechapter{\@Alph\c@chapter}}
\newcommand\appendixend{% removing \appendixname and numbering
	\gdef\@chapapp{}%
	\renewcommand\chapternamethechapter[1]{##1}%
	\gdef\thechapter{}}
%\setlength\arraycolsep{5\p@}
%\setlength\tabcolsep{6\p@}
%\setlength\arrayrulewidth{.4\p@}
%\setlength\doublerulesep{2\p@}
%\setlength\tabbingsep{\labelsep}
%\skip\@mpfootins = \skip\footins
%\setlength\fboxsep{3\p@}
%\setlength\fboxrule{.4\p@}
%\@addtoreset {equation}{chapter}
\renewcommand\theequation	% 图、表、公式等，一律用阿拉伯数字按章顺序编号
  {\ifnum \c@chapter>\z@ \thechapter-\fi \@arabic\c@equation}
%\newcounter{figure}[chapter]
\renewcommand \thefigure	% 如图1-1、图2-2，表1-1、表2-1，公式（1-1）等
     {\ifnum \c@chapter>\z@ \thechapter-\fi \@arabic\c@figure}
%\def\fps@figure{tbp}
%\def\ftype@figure{1}
%\def\ext@figure{lof}
%\def\fnum@figure{\figurename\nobreakspace\thefigure}
%\newenvironment{figure}
%               {\@float{figure}}
%               {\end@float}
%\newenvironment{figure*}
%               {\@dblfloat{figure}}
%               {\end@dblfloat}
%\newcounter{table}[chapter]
\renewcommand \thetable
     {\ifnum \c@chapter>\z@ \thechapter-\fi \@arabic\c@table}
%\def\fps@table{tbp}
%\def\ftype@table{2}
%\def\ext@table{lot}
%\def\fnum@table{\tablename\nobreakspace\thetable}
%\newenvironment{table}
%               {\@float{table}}
%               {\end@float}
%\newenvironment{table*}
%               {\@dblfloat{table}}
%               {\end@dblfloat}
%\newlength\abovecaptionskip
%\newlength\belowcaptionskip
\setlength\abovecaptionskip{2\p@}
\setlength\belowcaptionskip{2\p@}
%%%% If not want to centering some short caption, uncomment below.
%%%%	or if want to centering long caption as well, uncomment below.
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
	\begin{center}
	#1\hspace{1em}#2
	\end{center}
	%图序及图名置于图的下方，居中排列
	%表序及表名置于表的上方，居中排列
  %\sbox\@tempboxa{#1: #2}%
  %\ifdim \wd\@tempboxa >\hsize
  %  #1: #2\par
  %\else
  %  \global \@minipagefalse
  %  \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  %\fi
  \vskip\belowcaptionskip}
%\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
%\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
%\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
%\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
%\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
%\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
%\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
%\DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
%\DeclareRobustCommand*\mit{\@fontswitch\relax\mathnormal}
%\newcommand\@pnumwidth{1.55em}
%\newcommand\@tocrmarg{2.55em}
%\newcommand\@dotsep{4.5}
%\setcounter{tocdepth}{2}
\renewcommand\tableofcontents{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\contentsname
        \@mkboth{%
           \MakeUppercase\contentsname}{\MakeUppercase\contentsname}}%
	\addcontentsline{toc}{chapter}{\heiti \contentsname}%
	% It seems strange to \addcontentsline for the toc itself, but
	%	the LS of SYsU asked me to do that, and it is really
	%	convinient when read pdf with nevigation panel.
    \@starttoc{toc}%
    \if@restonecol\twocolumn\fi
    }
%\newcommand*\l@part[2]{%
%  \ifnum \c@tocdepth >-2\relax
%    \addpenalty{-\@highpenalty}%
%    \addvspace{2.25em \@plus\p@}%
%    \setlength\@tempdima{3em}%
%    \begingroup
%      \parindent \z@ \rightskip \@pnumwidth
%      \parfillskip -\@pnumwidth
%      {\leavevmode
%       \large \bfseries #1\hfil \hb@xt@\@pnumwidth{\hss #2}}\par
%       \nobreak
%         \global\@nobreaktrue
%         \everypar{\global\@nobreakfalse\everypar{}}%
%    \endgroup
%  \fi}
\renewcommand*\l@chapter[2]{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \vskip 1.0em \@plus\p@
    %\setlength\@tempdima{1.5em}%
    \setlength\@tempdima{4.5em}% the space after chapter order in toc
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \bfseries
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      %#1\nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      {#1}\nobreak
     \leaders\hbox{$\m@th
        \mkern \@dotsep mu\hbox{.}\mkern \@dotsep
        mu$}\hfill
	\nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi}
%\newcommand*\l@section{\@dottedtocline{1}{1.5em}{2.3em}}
%\newcommand*\l@subsection{\@dottedtocline{2}{3.8em}{3.2em}}
%\newcommand*\l@subsubsection{\@dottedtocline{3}{7.0em}{4.1em}}
%\newcommand*\l@paragraph{\@dottedtocline{4}{10em}{5em}}
%\newcommand*\l@subparagraph{\@dottedtocline{5}{12em}{6em}}
%\newcommand\listoffigures{%
%    \if@twocolumn
%      \@restonecoltrue\onecolumn
%    \else
%      \@restonecolfalse
%    \fi
%    \chapter*{\listfigurename}%
%      \@mkboth{\MakeUppercase\listfigurename}%
%              {\MakeUppercase\listfigurename}%
%    \@starttoc{lof}%
%    \if@restonecol\twocolumn\fi
%    }
%\newcommand*\l@figure{\@dottedtocline{1}{1.5em}{2.3em}}
%\newcommand\listoftables{%
%    \if@twocolumn
%      \@restonecoltrue\onecolumn
%    \else
%      \@restonecolfalse
%    \fi
%    \chapter*{\listtablename}%
%      \@mkboth{%
%          \MakeUppercase\listtablename}%
%         {\MakeUppercase\listtablename}%
%    \@starttoc{lot}%
%    \if@restonecol\twocolumn\fi
%    }
%\let\l@table\l@figure
%\newdimen\bibindent
%\setlength\bibindent{1.5em}
%\newenvironment{thebibliography}[1]
%     {\chapter*{\bibname}%
%      \@mkboth{\MakeUppercase\bibname}{\MakeUppercase\bibname}%
%      \list{\@biblabel{\@arabic\c@enumiv}}%
%           {\settowidth\labelwidth{\@biblabel{#1}}%
%            \leftmargin\labelwidth
%            \advance\leftmargin\labelsep
%            \@openbib@code
%            \usecounter{enumiv}%
%            \let\p@enumiv\@empty
%            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
%      \sloppy
%      \clubpenalty4000
%      \@clubpenalty \clubpenalty
%      \widowpenalty4000%
%      \sfcode`\.\@m}
%     {\def\@noitemerr
%       {\@latex@warning{Empty `thebibliography' environment}}%
%      \endlist}
%\newcommand\newblock{\hskip .11em\@plus.33em\@minus.07em}
%\let\@openbib@code\@empty
%\newenvironment{theindex}
%               {\if@twocolumn
%                  \@restonecolfalse
%                \else
%                  \@restonecoltrue
%                \fi
%                \twocolumn[\@makeschapterhead{\indexname}]%
%                \@mkboth{\MakeUppercase\indexname}%
%                        {\MakeUppercase\indexname}%
%                \thispagestyle{plain}\parindent\z@
%                \parskip\z@ \@plus .3\p@\relax
%                \columnseprule \z@
%                \columnsep 35\p@
%                \let\item\@idxitem}
%               {\if@restonecol\onecolumn\else\clearpage\fi}
%\newcommand\@idxitem{\par\hangindent 40\p@}
%\newcommand\subitem{\@idxitem \hspace*{20\p@}}
%\newcommand\subsubitem{\@idxitem \hspace*{30\p@}}
%\newcommand\indexspace{\par \vskip 10\p@ \@plus5\p@ \@minus3\p@\relax}
%\renewcommand\footnoterule{%
%  \kern-3\p@
%  \hrule\@width.4\columnwidth
%  \kern2.6\p@}
%\@addtoreset{footnote}{chapter}
%\newcommand\@makefntext[1]{%
%    \parindent 1em%
%    \noindent
%    \hb@xt@1.8em{\hss\@makefnmark}#1}
%\renewcommand\contentsname{目录}
%\renewcommand\listfigurename{插图清单}
%\renewcommand\listtablename{附表清单}
%\renewcommand\bibname{参考文献}	% GB7714《文后参考文献著录规则》
%\renewcommand\indexname{索引}
%\renewcommand\figurename{图}
%\renewcommand\tablename{表}
%\renewcommand\partname{部分}
%\renewcommand\chaptername{章}
%\renewcommand\appendixname{附录}
%\newcommand\abstractname{摘要}
%%%% inspired by `UTF8/zh-Hans.cpx' in CJKcaption
%%%% Chinese thesis
%\newcommand{\keywordname}{关键词}
%\def\today{\number\year~年~\number\month~月~\number\day~日}
%\setlength\columnsep{10\p@}
%\setlength\columnseprule{0\p@}
%\pagestyle{headings}
%\pagenumbering{arabic}
%\if@twoside
%\else
%  \raggedbottom
%\fi
%\if@twocolumn
%  \twocolumn
%  \sloppy
%  \flushbottom
%\else
%  \onecolumn
%\fi
%\endinput
%%
%% End of file `book.cls'.

% modified from latex.ltx and bk12.clo:
\def\author#1#2{\gdef\@author{#1} \gdef\@authorenglish{#2}}
\def\title#1#2{\gdef\@title{#1} \gdef\@titleenglish{#2}}
\def\major#1#2{\gdef\@major{#1} \gdef\@majorenglish{#2}}
\def\minor#1{\gdef\@minor{#1}}
\def\school#1{\gdef\@school{#1}}
\def\supervisor#1#2{\gdef\@supervisor{#1} \gdef\@supervisorenglish{#2}}
\def \@floatboxreset {%
        \reset@font
	\small\sffamily\centering%\large
        \@setminipage
}
\def\captionce{%
   \ifx\@captype\@undefined
     \@latex@error{\noexpand\captionce outside float}\@ehd
     \expandafter\@gobble
   \else
     \refstepcounter\@captype
     \expandafter\@firstofone
   \fi
   {\@dblarg{\@captionce\@captype}}%
}
\long\def\@captionce#1[#2]#3#4{%
  \par
  \addcontentsline{\csname ext@#1\endcsname}{#1}%
    {\protect\numberline{\csname the#1\endcsname}{\ignorespaces #2}}%
  \begingroup
    \@parboxrestore
    \if@minipage
      \@setminipage
    \fi
    %\large
	\@makecaption{\csname fnum@#1\endcsname}{\ignorespaces #3}\vskip-24\p@
	\@makecaption{\csname fnum@#1english\endcsname}{\ignorespaces #4}\par
  \endgroup}
\def\fnum@figureenglish{\figurenameenglish\nobreakspace\thefigure}
\def\fnum@tableenglish{\tablenameenglish\nobreakspace\thetable}
\newcommand\figurenameenglish{Figure}
\newcommand\tablenameenglish{Table}
\newcommand{\abstractnameenglish}{ABSTRACT}
\newcommand{\keywordnameenglish}{Key Words}
\newcommand{\thesisnameenglish}{}	% No verbose
%\newcommand{\thesisnameenglish}{%
%	\noindent A Dissertation\\
%	submitted in partial fulfillment of the requirements\\
%	for the degree of Doctor of Philosophy in\\
%	Sun Yat-Sen University}
%\newcommand{\detailnameenglish}{
%	\begin{tabular}{rl}
%	\vspace{3mm}
%	Marjor:		&\@majorenglish\\
%	\vspace{3mm}
%	Name:		&\@authorenglish\\
%%	\vspace{3mm}
%%	Supervisor:	&\@supervisorenglish\\
%	\end{tabular}\par
%}
\newenvironment{abstract}[1][]{% the option can only be 'english'
	\cleardoublepage\thispagestyle{plain}%
	%{\noindent\csname thesisname#1\endcsname\par}
	\begin{center}
	{\sffamily\bfseries\Large\csname @title#1\endcsname\par}\vspace{2ex}
	\csname detailname#1\endcsname
	\end{center}
	\vspace{2ex}%
	\renewcommand{\parindent@section}{0pt}% not indent \abstractname
	
	\section*{\Large\csname abstractname#1\endcsname}
	\addcontentsline{toc}{chapter}{\csname abstractname#1\endcsname}%
	
	% \addcontentsline should be here so hyperref has a destination.
	}{% \setcounter{chapter}{0} not needed
	\cleardoublepage}
\newcommand{\keyword}[1]{%
	\vspace{\fill}
	\textbf{\keywordname:} #1}
\newcommand{\keywordenglish}[1]{%
	\vspace{\fill}
	\textbf{\keywordnameenglish:} #1}

	\newenvironment{abstract2}[1][]{% the option can only be 'english'
	\cleardoublepage\thispagestyle{plain}%
	%{\noindent\csname thesisname#1\endcsname\par}
	\begin{center}
	{\sffamily\bfseries\Large\csname @title#1\endcsname\par}\vspace{2ex}
	\csname detailname#1\endcsname
	\end{center}
	\vspace{2ex}%
	\renewcommand{\parindent@section}{0pt}% not indent \abstractname
	
	\section*{\heiti\centering\LARGE\csname abstractname#1\endcsname}
	\addcontentsline{toc}{chapter}{\heiti\csname abstractname#1\endcsname}%
	
	% \addcontentsline should be here so hyperref has a destination.
	}{% \setcounter{chapter}{0} not needed
	\cleardoublepage}

	\newenvironment{abstract3}[1][]{% the option can only be 'english'
	\cleardoublepage\thispagestyle{plain}%
	%{\noindent\csname thesisname#1\endcsname\par}
	\begin{center}
	{\sffamily\bfseries\Large\csname @title#1\endcsname\par}\vspace{2ex}
	\csname detailname#1\endcsname
	\end{center}
	\vspace{2ex}%
	}{% \setcounter{chapter}{0} not needed
	\cleardoublepage}

%%%% Style: this part is the real requirement of SYsU
% copied from indentfirst.sty
\linespread{1.5} % 行距以版面清晰、容易辨识和阅读为原则
% 40 fullwidth chars or 80 letters is best, but too wide for a4papaer
%	每一面的上方（天头）和左侧（订口）分别留边25mm以上，
%	下方（地脚）和右侧（切口）应分别留边20mm以上
\setlength{\topmargin}{0pt}
\setlength{\textheight}{\paperheight}
\addtolength{\textheight}{-1in}
\addtolength{\textheight}{-20mm}
\addtolength{\textheight}{-\topmargin}
\addtolength{\textheight}{-\headheight}
\addtolength{\textheight}{-\headsep}
\addtolength{\textheight}{-\footskip}	% \textheight~=227mm
\setlength{\oddsidemargin}{0pt}
\setlength{\evensidemargin}{20pt}
\setlength{\textwidth}{\paperwidth}
\setlength{\marginparwidth}{1in}
\addtolength{\textwidth}{-1in}	% 1in=2.54cm
\addtolength{\textwidth}{-\marginparwidth}
\addtolength{\textwidth}{-20pt}	% \textwidth~=165mm
% 20pt==\marginparsep + gap between margin notes and paper's right side
% 章的题名建议采用小二号黑体，居中; \Large+ 18pt
\newcommand{\size@chapter}{\fontsize{18pt}\baselineskip\selectfont}
% 节的题名建议采用小三号宋体，加粗，左起空两格; \large+ 15pt
\newcommand{\size@section}{\fontsize{14pt}\baselineskip\selectfont}
% 文章段落内容建议采用小四号宋体; \large 12pt
%\newcommand{\size@paragraph}{12}	
% 专业的书籍都是用缩进并且通常不用空行来分离段落
\setlength{\parindent}{24pt}	% two chinese char width
% 节左边空二格
\newcommand{\parindent@section}{24pt}	% two chinese char width
% 章上方空一行
%\newcommand{\skip@beforechapter}{1\baselineskip \@plus 0ex \@minus 0ex}
\newcommand{\skip@beforechapter}{1\baselineskip}
% 章节题目之间空两行
%\newcommand{\skip@afterchapter}{2\baselineskip \@plus 0.2ex \@minus 0.1ex}
\newcommand{\skip@afterchapter}{2\baselineskip}
% 节题段内之间空一行
%\newcommand{\skip@beforesection}{0.5\baselineskip \@plus -0.3ex \@minus -0.1ex}
%\newcommand{\skip@aftersection}{0.5\baselineskip \@plus -0.3ex \@minus -0.1ex}
\newcommand{\skip@beforesection}{-3.5ex \@plus -1ex \@minus -.2ex}
\newcommand{\skip@aftersection}{1.5ex \@plus .2ex}
\newcommand{\skip@beforeparagraph}{3.25ex \@plus1ex \@minus.2ex}
\newcommand{\skip@afterparagraph}{-1em}

%%%% Special command for Chinese, especially for sysuthesis.
\newcommand\chapternamethechapter[1]{\heiti\CJKorder~#1~\@chapapp}
% not trivial to make the cls compatible with latex/dvips/ps2pdf and pdfLaTeX
%\RequirePackage{CJKutf8}
%\begin{CJK*}{UTF8}{gbsn}
\makeatletter
\newcommand{\CJKorder}{\heiti 第}	% for ordinal number
\newcommand{\CJK@Doctor}{博士}
\newcommand{\CJK@Master}{硕士}
\newcommand{\CJK@Bachelor}{本科}
\newcommand{\CJK@Degree}{学位}
\newcommand{\CJK@Graduate}{毕业}
\newcommand{\CJK@SYsU}{中山大学}
%\newcommand{\thesisname}{\CJK@SYsU\@degree\@thesis{}论文}
\newcommand{\thesisname}{ 《数据挖掘导论》大作业}
%\newcommand{\detailname}{
%	\begin{tabular}{r@{}l}
%	\vspace{4mm}
%	专\ 业\ 名\ 称：		&~\@major\\
%	\vspace{4mm}
%	姓名：	&~\@author\\
%%	\vspace{4mm}
%%	导\ 师\ 姓\ 名：	&~\@supervisor\\
%	\end{tabular}\par
%}
\newcommand{\sysufig}{
	\begin{figure}[!t]
		\includegraphics[]{images/sysufig.png}
	\end{figure}
}
\newcommand\maketitlepage[1][%
	\begin{flushleft}
	\sysufig
	\begin{tabular}{rl@{}l}
	答辩委员会	&（签名）	&\\
	主席：		&	&\\
	\\
	委员：		&	&\\
	\\
			&	&\\
			\\
			&	&\\
			\\
			&	&
	\end{tabular}
	\end{flushleft}]{%
	\begin{titlepage}
	\large%
	\newcommand\semiHuge{\@setfontsize\semiHuge{24.72}{27.38}}
	\begin{center}
	{\sffamily\Huge\thesisname\par}\vspace{3ex}
	{\sffamily\bfseries\semiHuge\@title\par}\vspace{3ex}
	{\sffamily\bfseries\Large\@titleenglish\par}\vspace{5ex}
%	\begin{tabular}{rl}
%	专\ 业\ 名\ 称：		&\@major\\
%	姓名：	&\@author\\
%%	导\ 师\ 姓\ 名：	&\@supervisor\\
%	
%	%研究方向：	&\@minor\\
%	%院、系（所）：	&\@school\\
%	
%	\end{tabular}\par
	\end{center}\vspace{1ex}
	#1%
	\vspace{\stretch{5}}
	\begin{center}
	\sffamily
	\CJK@SYsU\@school$\bullet{}$中国广州\\
	\@date\par
	\end{center}
	\end{titlepage}
}
\renewcommand\contentsname{目录}
\renewcommand\listfigurename{插图清单}
\renewcommand\listtablename{附表清单}
\renewcommand\bibname{参考文献}	% GB7714《文后参考文献著录规则》
\renewcommand\indexname{索引}
\renewcommand\figurename{图}
\renewcommand\tablename{表}
\renewcommand\partname{部分}
\renewcommand\chaptername{\heiti 章}
\renewcommand\appendixname{附录}
\newcommand\abstractname{\heiti 摘\ \ 要}
%%%% inspired by `UTF8/zh-Hans.cpx' in CJKcaption
%%%% Chinese thesis
\newcommand{\keywordname}{\heiti 关键词}
\def\today{\number\year~年~\number\month~月~\number\day~日}
\makeatother
%\end{CJK*}

%\AtEndOfClass{}
%\AtEndOfPackage{}
%\AtBeginDocument{%
%	\if@filesw
%		\immediate\write\@mainaux{\string\begin{CJK*}{UTF8}{gbsn}}%
%		\immediate\write\@mainaux{\string\makeatletter}%
%	\fi
%	\begin{CJK*}{UTF8}{gbsn}
%	\CJKcaption{zh-Hans}% Similar to KOMA-script
%}
%\AtEndDocument{%
%	\clearpage% \cleardoublepage for two-column printing
%	%% avoid trouble when latex writes header, footer, or index of a
%	%% page containing CJK after closing the CJK environment.
%	%% See `/usr/share/doc/latex-cjk-common/CJK.txt.gz'.
%	\end{CJK*}
%	\if@filesw
%		\immediate\write\@mainaux{\string\end{CJK*}}%
%	\fi
%}

\endinput
%%
%% End of file `sysuthesis.cls'.
% vim: textwidth=72 : fileencoding=utf8 :
