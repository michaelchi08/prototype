tate-space 
representation of a linear system is written in the following form:


\begin{align}
	\dot{x}(t) &= A(t) x(t) + B(t) u(t) \\
	y(t) &= C(t) x(t) + D(t) u(t)
\end{align}

Where:

\begin{itemize}
	\vspace{-0.4cm}
	\setlength{\itemsep}{0pt}
	\setlength{\parskip}{0pt}
	\setlength{\parsep}{0pt}
	
	\item{$x$ is the state vector}
	\item{$y$ is the output vector}
	\item{$u$ is the input or control vector}
	\item{$A$ is the state matrix}
	\item{$B$ is the input matrix}
	\item{$C$ is the output matrix}
	\item{$D$ is the feedforward matrix}
\end{itemize}

For the omni-directional robot we define the state vector $x$, output vector 
$y$ and input vector $u$ as:

\begin{minipage}{0.3\linewidth} 
	\begin{equation} 
		x =
		\begin{bmatrix}
			x \\
			y \\
			\theta 
		\end{bmatrix}
	\end{equation} 
\end{minipage} 
\hspace{0.5cm} 
\begin{minipage}{0.3\linewidth} 
	\begin{equation} 
		y =
		\begin{bmatrix}
			x \\
			y \\
			\theta 
		\end{bmatrix}
	\end{equation} 
\end{minipage}
\hspace{0.5cm} 
\begin{minipage}{0.3\linewidth} 
	\begin{equation} 
		u =
		\begin{bmatrix}
			\omega_{1} \\
			\omega_{2} \\
			\omega_{3} 
		\end{bmatrix}
	\end{equation} 
\end{minipage}

Where our state vector equation is:

\begin{align}
	\dot{x}(t) &= A(t) x(t) + B(t) u(t) \\
	\dot{x}(t) 
	&= 
	\begin{bmatrix}
		0 \\
		0 \\
		0
	\end{bmatrix}
	\begin{bmatrix}
		x \\
		y \\
		\theta 
	\end{bmatrix}
	+
	\begin{bmatrix}
		0 \\
		0 \\
		0
	\end{bmatrix}
	u_{t}
\end{align}
