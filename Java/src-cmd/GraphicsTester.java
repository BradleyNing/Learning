import java.awt.*;
import javax.swing.*;

public class GraphicsTester extends JFrame {

	public GraphicsTester ()
	{ 
		super("demo font, color, draw");
		setVisible( true ); //show frame
		setSize( 480, 250 ); //set window size
	}

	public void paint( Graphics g ) 
	{
		super.paint( g );
		g.setFont( new Font( "SansSerif", Font.BOLD, 12 ) );
		g.setColor(Color.blue); //set color
		g.drawString("Font ScanSerif, bold, 12, blue",20,50);
		g.setFont( new Font( "Serif", Font.ITALIC, 14 ) );
		g.setColor(new Color(255,0,0));
		g.drawString( " Font Serif, italic, 14, red", 250, 50 );
		g.drawLine(20,60,460,60); //draw line
	}

	public static void main(String[] args)
	{
		GraphicsTester application = new GraphicsTester();
		application.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}
}